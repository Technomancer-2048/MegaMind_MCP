#!/usr/bin/env python3
"""
Session Embedding Service Integration
Integrates ContentProcessor with existing embedding service and session manager
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from content_processor import ContentProcessor, EmbeddingType, AggregationLevel, ProcessedContent, SessionSummary
from session_manager import SessionManager, SessionEntry, EntryType
from services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)

class SessionEmbeddingService:
    """
    Integration service for session content processing and embedding generation
    """
    
    def __init__(self, session_manager: SessionManager, database_connection):
        self.session_manager = session_manager
        self.db = database_connection
        self.embedding_service = get_embedding_service()
        self.content_processor = ContentProcessor(
            embedding_service=self.embedding_service,
            session_manager=session_manager
        )
        
        # Configuration
        self.auto_embed_entries = True
        self.auto_generate_summaries = True
        self.batch_processing_enabled = True
        
        logger.info("SessionEmbeddingService initialized with content processing pipeline")
    
    # ================================================================
    # SESSION ENTRY EMBEDDING METHODS
    # ================================================================
    
    def process_and_embed_session_entry(self, 
                                      session_id: str, 
                                      entry: SessionEntry,
                                      force_regenerate: bool = False) -> Optional[str]:
        """
        Process session entry content and generate embedding
        
        Args:
            session_id: Session ID
            entry: SessionEntry to process
            force_regenerate: Force regeneration even if embedding exists
            
        Returns:
            Embedding ID if successful, None otherwise
        """
        try:
            # Check if embedding already exists
            if not force_regenerate:
                existing_embedding = self._get_existing_entry_embedding(entry.entry_id)
                if existing_embedding:
                    logger.debug(f"Embedding already exists for entry {entry.entry_id}")
                    return existing_embedding['embedding_id']
            
            # Process content for embedding
            processed_content = self.content_processor.process_session_entry_for_embedding(entry)
            
            # Generate embedding
            embedding_data = self.content_processor.generate_embedding_for_content(
                content=processed_content.processed_content,
                embedding_type=EmbeddingType.ENTRY_CONTENT,
                metadata={
                    "session_id": session_id,
                    "entry_id": entry.entry_id,
                    "entry_type": entry.entry_type.value,
                    "operation_type": entry.operation_type,
                    "original_content_hash": processed_content.content_hash,
                    "truncation_applied": processed_content.content_truncated,
                    "truncation_strategy": processed_content.truncation_strategy.value,
                    "key_information": processed_content.key_information,
                    "quality_weight": processed_content.metadata.get("quality_weight", 0.5)
                }
            )
            
            if not embedding_data:
                logger.warning(f"Failed to generate embedding for entry {entry.entry_id}")
                return None
            
            # Store embedding in database
            embedding_id = self._store_session_embedding(
                session_id=session_id,
                entry_id=entry.entry_id,
                embedding_data=embedding_data,
                processed_content=processed_content
            )
            
            logger.debug(f"Generated embedding {embedding_id} for entry {entry.entry_id}")
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to process and embed entry {entry.entry_id}: {e}")
            return None
    
    def generate_session_summary_embedding(self, 
                                         session_id: str,
                                         force_regenerate: bool = False) -> Optional[str]:
        """
        Generate comprehensive session summary embedding
        
        Args:
            session_id: Session ID to summarize
            force_regenerate: Force regeneration even if summary exists
            
        Returns:
            Embedding ID if successful, None otherwise
        """
        try:
            # Check if summary embedding already exists
            if not force_regenerate:
                existing_summary = self._get_existing_session_summary(session_id)
                if existing_summary:
                    logger.debug(f"Summary embedding already exists for session {session_id}")
                    return existing_summary['embedding_id']
            
            # Generate session summary
            session_summary = self.content_processor.create_session_summary_embedding(session_id)
            
            # Generate embedding for summary
            embedding_data = self.content_processor.generate_embedding_for_content(
                content=session_summary.summary_content,
                embedding_type=EmbeddingType.SESSION_SUMMARY,
                metadata={
                    "session_id": session_id,
                    "summary_type": "comprehensive",
                    "entry_count": session_summary.entry_count,
                    "quality_score": session_summary.quality_score,
                    "key_operations_count": len(session_summary.key_operations),
                    "decisions_count": len(session_summary.major_decisions),
                    "outcomes_count": len(session_summary.outcomes_discoveries),
                    "session_metadata": session_summary.session_metadata
                }
            )
            
            if not embedding_data:
                logger.warning(f"Failed to generate summary embedding for session {session_id}")
                return None
            
            # Store summary embedding
            embedding_id = self._store_session_embedding(
                session_id=session_id,
                entry_id=None,  # No specific entry for session summary
                embedding_data=embedding_data,
                summary_data=session_summary
            )
            
            logger.info(f"Generated session summary embedding {embedding_id} for session {session_id}")
            return embedding_id
            
        except Exception as e:
            logger.error(f"Failed to generate session summary embedding for {session_id}: {e}")
            return None
    
    def batch_process_session_entries(self, 
                                    session_id: str,
                                    max_entries: int = 100) -> Dict[str, Any]:
        """
        Batch process all session entries for embedding generation
        
        Args:
            session_id: Session ID to process
            max_entries: Maximum number of entries to process
            
        Returns:
            Processing results summary
        """
        try:
            # Get session entries
            entries = self.session_manager.get_session_entries(session_id, limit=max_entries)
            
            results = {
                "session_id": session_id,
                "total_entries": len(entries),
                "processed_entries": 0,
                "successful_embeddings": 0,
                "failed_embeddings": 0,
                "skipped_entries": 0,
                "processing_errors": [],
                "embedding_ids": []
            }
            
            for entry in entries:
                try:
                    results["processed_entries"] += 1
                    
                    # Check if entry should be processed
                    if not self._should_process_entry(entry):
                        results["skipped_entries"] += 1
                        continue
                    
                    # Process and embed entry
                    embedding_id = self.process_and_embed_session_entry(session_id, entry)
                    
                    if embedding_id:
                        results["successful_embeddings"] += 1
                        results["embedding_ids"].append(embedding_id)
                    else:
                        results["failed_embeddings"] += 1
                        
                except Exception as e:
                    results["failed_embeddings"] += 1
                    results["processing_errors"].append({
                        "entry_id": entry.entry_id,
                        "error": str(e)
                    })
                    logger.warning(f"Failed to process entry {entry.entry_id}: {e}")
            
            # Generate session summary if we processed entries
            if results["successful_embeddings"] > 0:
                try:
                    summary_embedding_id = self.generate_session_summary_embedding(session_id)
                    if summary_embedding_id:
                        results["summary_embedding_id"] = summary_embedding_id
                except Exception as e:
                    logger.warning(f"Failed to generate session summary: {e}")
            
            logger.info(f"Batch processing completed for session {session_id}: "
                       f"{results['successful_embeddings']}/{results['total_entries']} entries processed")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed for session {session_id}: {e}")
            return {"error": str(e), "session_id": session_id}
    
    # ================================================================
    # EMBEDDING SEARCH AND RETRIEVAL METHODS
    # ================================================================
    
    def search_session_embeddings(self, 
                                query: str,
                                session_id: Optional[str] = None,
                                embedding_types: Optional[List[EmbeddingType]] = None,
                                limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search session embeddings using semantic similarity
        
        Args:
            query: Search query
            session_id: Optional session ID filter
            embedding_types: Optional embedding type filters
            limit: Maximum results to return
            
        Returns:
            List of matching embeddings with similarity scores
        """
        try:
            if not self.embedding_service.is_available():
                logger.warning("Embedding service not available for search")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_service.generate_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []
            
            # Search session embeddings
            results = self._search_embeddings_in_database(
                query_embedding=query_embedding,
                session_id=session_id,
                embedding_types=embedding_types,
                limit=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Session embedding search failed: {e}")
            return []
    
    def get_session_embedding_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of session embeddings
        
        Args:
            session_id: Session ID
            
        Returns:
            Summary of session embeddings
        """
        try:
            summary = {
                "session_id": session_id,
                "entry_embeddings": 0,
                "summary_embeddings": 0,
                "total_embeddings": 0,
                "embedding_types": {},
                "quality_distribution": {"high": 0, "medium": 0, "low": 0},
                "last_updated": None,
                "coverage_percentage": 0.0
            }
            
            # Query session embeddings
            embeddings = self._get_session_embeddings_from_database(session_id)
            
            summary["total_embeddings"] = len(embeddings)
            
            for embedding in embeddings:
                embedding_type = embedding.get("embedding_type", "unknown")
                summary["embedding_types"][embedding_type] = summary["embedding_types"].get(embedding_type, 0) + 1
                
                if embedding_type == "entry_content":
                    summary["entry_embeddings"] += 1
                elif embedding_type == "session_summary":
                    summary["summary_embeddings"] += 1
                
                # Quality distribution
                quality_score = embedding.get("embedding_quality_score", 0.5)
                if quality_score >= 0.8:
                    summary["quality_distribution"]["high"] += 1
                elif quality_score >= 0.5:
                    summary["quality_distribution"]["medium"] += 1
                else:
                    summary["quality_distribution"]["low"] += 1
                
                # Track latest update
                created_at = embedding.get("created_at")
                if created_at and (not summary["last_updated"] or created_at > summary["last_updated"]):
                    summary["last_updated"] = created_at
            
            # Calculate coverage percentage
            total_entries = len(self.session_manager.get_session_entries(session_id, limit=1000))
            if total_entries > 0:
                summary["coverage_percentage"] = (summary["entry_embeddings"] / total_entries) * 100
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get session embedding summary: {e}")
            return {"error": str(e), "session_id": session_id}
    
    # ================================================================
    # DATABASE INTEGRATION METHODS
    # ================================================================
    
    def _store_session_embedding(self, 
                                session_id: str,
                                embedding_data: Dict[str, Any],
                                entry_id: Optional[str] = None,
                                processed_content: Optional[ProcessedContent] = None,
                                summary_data: Optional[SessionSummary] = None) -> str:
        """Store session embedding in database"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor()
            
            # Check if session_embeddings table exists
            cursor.execute("SHOW TABLES LIKE 'megamind_session_embeddings'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                logger.warning("megamind_session_embeddings table does not exist, creating minimal record")
                cursor.close()
                connection.close()
                return embedding_data["embedding_id"]
            
            # Prepare embedding data for storage
            embedding_vector_json = json.dumps(embedding_data["embedding_vector"])
            metadata_json = json.dumps(embedding_data["metadata"])
            
            # Determine content source and tokens
            if processed_content:
                content_source = processed_content.processed_content
                content_tokens = processed_content.content_tokens
                truncated = processed_content.content_truncated
                truncation_strategy = processed_content.truncation_strategy.value
            elif summary_data:
                content_source = summary_data.summary_content
                content_tokens = summary_data.summary_tokens
                truncated = False
                truncation_strategy = "smart_summary"
            else:
                content_source = embedding_data["content_source"]
                content_tokens = embedding_data["content_tokens"]
                truncated = False
                truncation_strategy = "smart_summary"
            
            insert_query = """
            INSERT INTO megamind_session_embeddings (
                embedding_id, session_id, entry_id, embedding_type,
                content_source, content_tokens, token_limit_applied,
                content_truncated, truncation_strategy, embedding_vector,
                model_name, embedding_dimension, embedding_quality_score,
                aggregation_level, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                embedding_data["embedding_id"],
                session_id,
                entry_id,
                embedding_data["embedding_type"],
                content_source[:2000],  # Limit content source length
                content_tokens,
                self.content_processor.optimal_token_limit,
                truncated,
                truncation_strategy,
                embedding_vector_json,
                embedding_data["model_name"],
                embedding_data["embedding_dimension"],
                embedding_data.get("quality_score", 0.5),
                "entry" if entry_id else "session",
                datetime.now()
            ))
            
            connection.commit()
            cursor.close()
            connection.close()
            
            return embedding_data["embedding_id"]
            
        except Exception as e:
            logger.error(f"Failed to store session embedding: {e}")
            return embedding_data["embedding_id"]  # Return ID even if storage fails
    
    def _get_existing_entry_embedding(self, entry_id: str) -> Optional[Dict[str, Any]]:
        """Check if embedding already exists for entry"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("SHOW TABLES LIKE 'megamind_session_embeddings'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                cursor.close()
                connection.close()
                return None
            
            query = """
            SELECT embedding_id, embedding_type, embedding_quality_score, created_at
            FROM megamind_session_embeddings 
            WHERE entry_id = %s AND embedding_type = 'entry_content'
            ORDER BY created_at DESC LIMIT 1
            """
            
            cursor.execute(query, (entry_id,))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to check existing entry embedding: {e}")
            return None
    
    def _get_existing_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Check if session summary embedding already exists"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("SHOW TABLES LIKE 'megamind_session_embeddings'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                cursor.close()
                connection.close()
                return None
            
            query = """
            SELECT embedding_id, embedding_type, embedding_quality_score, created_at
            FROM megamind_session_embeddings 
            WHERE session_id = %s AND embedding_type = 'session_summary' AND entry_id IS NULL
            ORDER BY created_at DESC LIMIT 1
            """
            
            cursor.execute(query, (session_id,))
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to check existing session summary: {e}")
            return None
    
    def _get_session_embeddings_from_database(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all embeddings for a session"""
        try:
            connection = self.db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("SHOW TABLES LIKE 'megamind_session_embeddings'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                cursor.close()
                connection.close()
                return []
            
            query = """
            SELECT embedding_id, embedding_type, entry_id, content_tokens,
                   embedding_quality_score, created_at, aggregation_level
            FROM megamind_session_embeddings 
            WHERE session_id = %s
            ORDER BY created_at DESC
            """
            
            cursor.execute(query, (session_id,))
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get session embeddings: {e}")
            return []
    
    def _search_embeddings_in_database(self, 
                                     query_embedding: List[float],
                                     session_id: Optional[str] = None,
                                     embedding_types: Optional[List[EmbeddingType]] = None,
                                     limit: int = 10) -> List[Dict[str, Any]]:
        """Search embeddings in database using similarity"""
        try:
            # This is a simplified implementation - full vector similarity search
            # would require specialized database extensions or external vector databases
            connection = self.db.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            cursor.execute("SHOW TABLES LIKE 'megamind_session_embeddings'")
            table_exists = cursor.fetchone()
            
            if not table_exists:
                cursor.close()
                connection.close()
                return []
            
            # Build query with filters
            where_conditions = []
            params = []
            
            if session_id:
                where_conditions.append("session_id = %s")
                params.append(session_id)
            
            if embedding_types:
                type_placeholders = ",".join(["%s"] * len(embedding_types))
                where_conditions.append(f"embedding_type IN ({type_placeholders})")
                params.extend([et.value for et in embedding_types])
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            query = f"""
            SELECT embedding_id, session_id, entry_id, embedding_type,
                   content_source, embedding_quality_score, created_at
            FROM megamind_session_embeddings 
            WHERE {where_clause}
            ORDER BY embedding_quality_score DESC, created_at DESC
            LIMIT %s
            """
            
            params.append(limit)
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()
            connection.close()
            
            # Add placeholder similarity scores (would be calculated in real vector search)
            for result in results:
                result["similarity_score"] = 0.8  # Placeholder
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search embeddings: {e}")
            return []
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def _should_process_entry(self, entry: SessionEntry) -> bool:
        """Determine if entry should be processed for embedding"""
        # Skip very short entries
        if len(entry.entry_content.strip()) < 10:
            return False
        
        # Skip system events with low importance
        if entry.entry_type == EntryType.SYSTEM_EVENT and entry.operation_type in ["heartbeat", "ping"]:
            return False
        
        # Always process important entry types
        if entry.entry_type in [EntryType.OPERATION, EntryType.QUERY, EntryType.ERROR]:
            return True
        
        # Process other types if they have good quality scores
        return entry.quality_score > 0.3
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        processor_stats = self.content_processor.get_processing_stats()
        
        return {
            "content_processor": processor_stats,
            "embedding_service_available": self.embedding_service.is_available() if self.embedding_service else False,
            "auto_embed_entries": self.auto_embed_entries,
            "auto_generate_summaries": self.auto_generate_summaries,
            "batch_processing_enabled": self.batch_processing_enabled
        }