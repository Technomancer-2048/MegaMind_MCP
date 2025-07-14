#!/usr/bin/env python3
"""
Content Processing Pipeline for Enhanced Session System
Implements token-aware content processing, summarization, and embedding generation
"""

import json
import logging
import re
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict

# Import session manager components
from session_manager import SessionEntry, EntryType, SessionMetadata

logger = logging.getLogger(__name__)

class TruncationStrategy(Enum):
    """Content truncation strategies"""
    HEAD = "head"              # Keep beginning
    TAIL = "tail"              # Keep end
    MIDDLE = "middle"          # Keep middle section
    SMART_SUMMARY = "smart_summary"  # Intelligent summarization

class EmbeddingType(Enum):
    """Types of embeddings to generate"""
    ENTRY_CONTENT = "entry_content"
    SESSION_SUMMARY = "session_summary"
    CONTEXT_WINDOW = "context_window"
    CONVERSATION_TURN = "conversation_turn"

class AggregationLevel(Enum):
    """Multi-level aggregation strategies"""
    ENTRY = "entry"
    TURN = "turn" 
    SESSION = "session"
    CROSS_SESSION = "cross_session"

@dataclass
class ProcessedContent:
    """Processed content structure"""
    original_content: str
    processed_content: str
    content_tokens: int
    content_truncated: bool
    truncation_strategy: TruncationStrategy
    content_hash: str
    key_information: List[str]
    metadata: Dict[str, Any]

@dataclass
class SessionSummary:
    """Session summary structure"""
    session_id: str
    summary_content: str
    summary_tokens: int
    key_operations: List[str]
    major_decisions: List[str]
    outcomes_discoveries: List[str]
    session_metadata: Dict[str, Any]
    created_at: datetime
    entry_count: int
    quality_score: float

@dataclass
class EmbeddingRequest:
    """Embedding generation request"""
    content_id: str
    content_source: str
    embedding_type: EmbeddingType
    content_tokens: int
    truncation_applied: bool
    aggregation_level: AggregationLevel
    metadata: Dict[str, Any]

class ContentProcessor:
    """
    Content processing pipeline with token-aware processing, summarization, and embedding generation
    """
    
    def __init__(self, embedding_service=None, session_manager=None):
        self.embedding_service = embedding_service
        self.session_manager = session_manager
        
        # Token management configuration
        self.optimal_token_limit = 128
        self.maximum_token_limit = 256
        self.token_estimation_ratio = 4  # 1 token ≈ 4 characters
        
        # Content processing configuration
        self.summary_key_operations_limit = 5
        self.summary_decisions_limit = 3
        self.summary_discoveries_limit = 3
        
        # Entry type processing weights
        self.entry_type_weights = {
            EntryType.QUERY: 0.7,
            EntryType.OPERATION: 1.0,
            EntryType.RESULT: 0.9,
            EntryType.CONTEXT_SWITCH: 0.5,
            EntryType.ERROR: 0.8,
            EntryType.SYSTEM_EVENT: 0.3
        }
        
        logger.info("ContentProcessor initialized with token-aware processing pipeline")
    
    # ================================================================
    # CORE CONTENT PROCESSING METHODS
    # ================================================================
    
    def process_session_entry_for_embedding(self, 
                                           entry: SessionEntry,
                                           target_token_limit: Optional[int] = None) -> ProcessedContent:
        """
        Process session entry content for embedding generation with token awareness
        
        Args:
            entry: SessionEntry object to process
            target_token_limit: Optional token limit override
            
        Returns:
            ProcessedContent with optimized content for embedding
        """
        try:
            target_limit = target_token_limit or self.optimal_token_limit
            
            # Estimate tokens in original content
            estimated_tokens = self._estimate_tokens(entry.entry_content)
            
            # Extract key information based on entry type
            key_info = self._extract_key_information(entry)
            
            # Apply content processing strategy
            if estimated_tokens <= target_limit:
                # Content fits within limit
                processed_content = self._enhance_content_context(entry.entry_content, key_info, entry.entry_type)
                truncation_strategy = TruncationStrategy.SMART_SUMMARY
                content_truncated = False
            else:
                # Content needs truncation
                processed_content, truncation_strategy = self._apply_truncation_strategy(
                    entry.entry_content, target_limit, entry.entry_type, key_info
                )
                content_truncated = True
            
            # Generate content hash
            content_hash = hashlib.md5(processed_content.encode('utf-8')).hexdigest()
            
            # Calculate final token count
            final_tokens = self._estimate_tokens(processed_content)
            
            return ProcessedContent(
                original_content=entry.entry_content,
                processed_content=processed_content,
                content_tokens=final_tokens,
                content_truncated=content_truncated,
                truncation_strategy=truncation_strategy,
                content_hash=content_hash,
                key_information=key_info,
                metadata={
                    "entry_type": entry.entry_type.value,
                    "operation_type": entry.operation_type,
                    "original_tokens": estimated_tokens,
                    "processing_timestamp": datetime.now().isoformat(),
                    "related_chunks": entry.related_chunk_ids or [],
                    "quality_weight": self.entry_type_weights.get(entry.entry_type, 0.5)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to process entry {entry.entry_id}: {e}")
            raise
    
    def create_session_summary_embedding(self, 
                                       session_id: str,
                                       target_token_limit: Optional[int] = None) -> SessionSummary:
        """
        Create comprehensive session summary for embedding generation
        
        Args:
            session_id: Session ID to summarize
            target_token_limit: Optional token limit override
            
        Returns:
            SessionSummary with optimized content for embedding
        """
        try:
            if not self.session_manager:
                raise ValueError("SessionManager required for session summary generation")
            
            target_limit = target_token_limit or self.optimal_token_limit
            
            # Retrieve session metadata and entries
            session = self.session_manager.get_session(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            entries = self.session_manager.get_session_entries(session_id, limit=1000)
            
            # Analyze and categorize entries
            analysis = self._analyze_session_entries(entries)
            
            # Generate structured summary
            summary_content = self._generate_structured_summary(
                session, analysis, target_limit
            )
            
            # Calculate quality score
            quality_score = self._calculate_summary_quality_score(analysis, len(entries))
            
            return SessionSummary(
                session_id=session_id,
                summary_content=summary_content,
                summary_tokens=self._estimate_tokens(summary_content),
                key_operations=analysis["key_operations"],
                major_decisions=analysis["major_decisions"],
                outcomes_discoveries=analysis["outcomes_discoveries"],
                session_metadata={
                    "session_name": session.session_name,
                    "realm_id": session.realm_id,
                    "priority": session.priority.value,
                    "state": session.session_state.value,
                    "total_entries": len(entries),
                    "entry_types": analysis["entry_type_counts"]
                },
                created_at=datetime.now(),
                entry_count=len(entries),
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.error(f"Failed to create session summary for {session_id}: {e}")
            raise
    
    def generate_embedding_for_content(self, 
                                     content: str,
                                     embedding_type: EmbeddingType,
                                     metadata: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Generate embedding for processed content using the embedding service
        
        Args:
            content: Processed content to embed
            embedding_type: Type of embedding to generate
            metadata: Additional metadata for embedding context
            
        Returns:
            Embedding data or None if service unavailable
        """
        try:
            if not self.embedding_service or not self.embedding_service.is_available():
                logger.warning("Embedding service not available")
                return None
            
            # Prepare realm context for enhanced embedding
            realm_context = metadata.get("realm_id") if metadata else None
            
            # Generate embedding
            embedding_vector = self.embedding_service.generate_embedding(
                content, realm_context=realm_context
            )
            
            if embedding_vector is None:
                logger.warning("Embedding generation returned None")
                return None
            
            # Package embedding data
            embedding_data = {
                "embedding_id": f"emb_{uuid.uuid4().hex[:12]}",
                "embedding_vector": embedding_vector,
                "embedding_type": embedding_type.value,
                "content_source": content,
                "content_tokens": self._estimate_tokens(content),
                "model_name": getattr(self.embedding_service, 'model_name', 'unknown'),
                "embedding_dimension": len(embedding_vector),
                "quality_score": self._calculate_embedding_quality_score(content, embedding_vector),
                "metadata": metadata or {},
                "created_at": datetime.now().isoformat()
            }
            
            return embedding_data
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    # ================================================================
    # CONTENT ANALYSIS AND EXTRACTION METHODS
    # ================================================================
    
    def _extract_key_information(self, entry: SessionEntry) -> List[str]:
        """Extract key information based on entry type"""
        key_info = []
        content = entry.entry_content.lower()
        
        try:
            if entry.entry_type == EntryType.QUERY:
                # Extract search terms, question keywords
                query_patterns = [
                    r"search for (\w+(?:\s+\w+){0,3})",
                    r"find (\w+(?:\s+\w+){0,3})",
                    r"how to (\w+(?:\s+\w+){0,3})",
                    r"what is (\w+(?:\s+\w+){0,3})",
                    r"where (\w+(?:\s+\w+){0,3})"
                ]
                for pattern in query_patterns:
                    matches = re.findall(pattern, content)
                    key_info.extend(matches)
            
            elif entry.entry_type == EntryType.OPERATION:
                # Extract operation types, targets, outcomes
                operation_patterns = [
                    r"created? (\w+(?:\s+\w+){0,2})",
                    r"updated? (\w+(?:\s+\w+){0,2})",
                    r"deleted? (\w+(?:\s+\w+){0,2})",
                    r"modified? (\w+(?:\s+\w+){0,2})",
                    r"executed? (\w+(?:\s+\w+){0,2})"
                ]
                for pattern in operation_patterns:
                    matches = re.findall(pattern, content)
                    key_info.extend(matches)
            
            elif entry.entry_type == EntryType.RESULT:
                # Extract outcomes, metrics, success indicators
                result_patterns = [
                    r"successfully (\w+(?:\s+\w+){0,2})",
                    r"failed to (\w+(?:\s+\w+){0,2})",
                    r"completed (\w+(?:\s+\w+){0,2})",
                    r"(\d+) (\w+) (\w+)",  # Numeric results
                    r"error: (\w+(?:\s+\w+){0,2})"
                ]
                for pattern in result_patterns:
                    matches = re.findall(pattern, content)
                    if isinstance(matches[0], tuple) if matches else False:
                        key_info.extend([" ".join(match) for match in matches])
                    else:
                        key_info.extend(matches)
            
            elif entry.entry_type == EntryType.ERROR:
                # Extract error types, causes, solutions
                error_patterns = [
                    r"error: (\w+(?:\s+\w+){0,3})",
                    r"failed (\w+(?:\s+\w+){0,3})",
                    r"exception: (\w+(?:\s+\w+){0,3})",
                    r"unable to (\w+(?:\s+\w+){0,3})"
                ]
                for pattern in error_patterns:
                    matches = re.findall(pattern, content)
                    key_info.extend(matches)
            
            # Add operation type if available
            if entry.operation_type:
                key_info.append(entry.operation_type)
            
            # Clean and deduplicate
            key_info = list(set([info.strip() for info in key_info if info.strip()]))
            
            return key_info[:10]  # Limit to top 10 key pieces of information
            
        except Exception as e:
            logger.warning(f"Failed to extract key information: {e}")
            return []
    
    def _analyze_session_entries(self, entries: List[SessionEntry]) -> Dict[str, Any]:
        """Analyze session entries to extract key patterns and information"""
        try:
            analysis = {
                "key_operations": [],
                "major_decisions": [],
                "outcomes_discoveries": [],
                "entry_type_counts": {},
                "operation_types": {},
                "error_patterns": [],
                "success_indicators": [],
                "timeline_markers": []
            }
            
            # Count entry types
            for entry in entries:
                entry_type = entry.entry_type.value
                analysis["entry_type_counts"][entry_type] = analysis["entry_type_counts"].get(entry_type, 0) + 1
                
                if entry.operation_type:
                    analysis["operation_types"][entry.operation_type] = analysis["operation_types"].get(entry.operation_type, 0) + 1
            
            # Extract key operations (high-impact operations)
            operation_entries = [e for e in entries if e.entry_type == EntryType.OPERATION]
            operation_entries.sort(key=lambda x: self.entry_type_weights.get(x.entry_type, 0.5), reverse=True)
            
            for entry in operation_entries[:self.summary_key_operations_limit]:
                key_info = self._extract_key_information(entry)
                if key_info:
                    operation_summary = f"{entry.operation_type or 'Operation'}: {', '.join(key_info[:3])}"
                    analysis["key_operations"].append(operation_summary)
            
            # Extract major decisions (context switches and significant operations)
            decision_entries = [e for e in entries if e.entry_type in [EntryType.CONTEXT_SWITCH, EntryType.OPERATION]]
            decision_entries.sort(key=lambda x: x.context_relevance_score, reverse=True)
            
            for entry in decision_entries[:self.summary_decisions_limit]:
                if "decision" in entry.entry_content.lower() or "chose" in entry.entry_content.lower():
                    decision_summary = entry.entry_content[:100] + "..." if len(entry.entry_content) > 100 else entry.entry_content
                    analysis["major_decisions"].append(decision_summary)
            
            # Extract outcomes and discoveries (results and successful operations)
            result_entries = [e for e in entries if e.entry_type == EntryType.RESULT and e.success_indicator]
            result_entries.sort(key=lambda x: x.quality_score, reverse=True)
            
            for entry in result_entries[:self.summary_discoveries_limit]:
                outcome_summary = entry.entry_content[:80] + "..." if len(entry.entry_content) > 80 else entry.entry_content
                analysis["outcomes_discoveries"].append(outcome_summary)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze session entries: {e}")
            return {"key_operations": [], "major_decisions": [], "outcomes_discoveries": [], "entry_type_counts": {}}
    
    def _generate_structured_summary(self, 
                                   session: SessionMetadata, 
                                   analysis: Dict[str, Any], 
                                   target_token_limit: int) -> str:
        """Generate structured session summary within token limits"""
        try:
            # Build summary components
            summary_parts = []
            
            # Session header
            header = f"Session: {session.session_name} (State: {session.session_state.value})"
            summary_parts.append(header)
            
            # Key operations
            if analysis["key_operations"]:
                operations_text = "Key Operations: " + "; ".join(analysis["key_operations"])
                summary_parts.append(operations_text)
            
            # Major decisions
            if analysis["major_decisions"]:
                decisions_text = "Decisions: " + "; ".join(analysis["major_decisions"])
                summary_parts.append(decisions_text)
            
            # Outcomes and discoveries
            if analysis["outcomes_discoveries"]:
                outcomes_text = "Outcomes: " + "; ".join(analysis["outcomes_discoveries"])
                summary_parts.append(outcomes_text)
            
            # Entry statistics
            total_entries = sum(analysis["entry_type_counts"].values())
            stats_text = f"Total Entries: {total_entries}"
            if analysis["entry_type_counts"]:
                top_types = sorted(analysis["entry_type_counts"].items(), key=lambda x: x[1], reverse=True)[:3]
                type_summary = ", ".join([f"{t}:{c}" for t, c in top_types])
                stats_text += f" ({type_summary})"
            summary_parts.append(stats_text)
            
            # Combine and check token limits
            full_summary = ". ".join(summary_parts)
            
            # Apply token limit
            if self._estimate_tokens(full_summary) > target_token_limit:
                full_summary = self._truncate_to_token_limit(full_summary, target_token_limit)
            
            return full_summary
            
        except Exception as e:
            logger.error(f"Failed to generate structured summary: {e}")
            return f"Session {session.session_id} summary unavailable"
    
    # ================================================================
    # TOKEN MANAGEMENT AND TRUNCATION METHODS
    # ================================================================
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using character-based approximation"""
        if not text:
            return 0
        return max(1, len(text) // self.token_estimation_ratio)
    
    def _apply_truncation_strategy(self, 
                                 content: str, 
                                 target_limit: int, 
                                 entry_type: EntryType,
                                 key_info: List[str]) -> Tuple[str, TruncationStrategy]:
        """Apply intelligent truncation strategy based on content type"""
        try:
            target_chars = target_limit * self.token_estimation_ratio
            
            if entry_type in [EntryType.ERROR, EntryType.RESULT]:
                # For errors and results, prefer tail (keep the conclusion)
                truncated = self._truncate_tail(content, target_chars)
                strategy = TruncationStrategy.TAIL
            elif entry_type == EntryType.QUERY:
                # For queries, prefer head (keep the question)
                truncated = self._truncate_head(content, target_chars)
                strategy = TruncationStrategy.HEAD
            else:
                # For operations and others, use smart summary
                truncated = self._smart_truncate_with_key_info(content, target_chars, key_info)
                strategy = TruncationStrategy.SMART_SUMMARY
            
            return truncated, strategy
            
        except Exception as e:
            logger.warning(f"Truncation failed, using simple head truncation: {e}")
            return content[:target_limit * self.token_estimation_ratio], TruncationStrategy.HEAD
    
    def _truncate_head(self, content: str, target_chars: int) -> str:
        """Keep the beginning of the content"""
        if len(content) <= target_chars:
            return content
        return content[:target_chars - 3] + "..."
    
    def _truncate_tail(self, content: str, target_chars: int) -> str:
        """Keep the end of the content"""
        if len(content) <= target_chars:
            return content
        return "..." + content[-(target_chars - 3):]
    
    def _smart_truncate_with_key_info(self, content: str, target_chars: int, key_info: List[str]) -> str:
        """Intelligent truncation preserving key information"""
        if len(content) <= target_chars:
            return content
        
        try:
            # Try to preserve sentences containing key information
            sentences = re.split(r'[.!?]+', content)
            key_sentences = []
            other_sentences = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                # Check if sentence contains key information
                contains_key_info = any(key.lower() in sentence.lower() for key in key_info)
                if contains_key_info:
                    key_sentences.append(sentence)
                else:
                    other_sentences.append(sentence)
            
            # Start with key sentences
            result = ". ".join(key_sentences)
            
            # Add other sentences if space allows
            for sentence in other_sentences:
                candidate = result + ". " + sentence if result else sentence
                if len(candidate) <= target_chars - 3:
                    result = candidate
                else:
                    break
            
            # If still too long, truncate
            if len(result) > target_chars:
                result = result[:target_chars - 3] + "..."
            
            return result if result else content[:target_chars - 3] + "..."
            
        except Exception as e:
            logger.warning(f"Smart truncation failed: {e}")
            return content[:target_chars - 3] + "..."
    
    def _truncate_to_token_limit(self, text: str, token_limit: int) -> str:
        """Truncate text to specific token limit"""
        char_limit = token_limit * self.token_estimation_ratio
        if len(text) <= char_limit:
            return text
        return text[:char_limit - 3] + "..."
    
    def _enhance_content_context(self, content: str, key_info: List[str], entry_type: EntryType) -> str:
        """Enhance content with context information for better embedding"""
        try:
            # Add entry type context prefix
            type_prefix = f"[{entry_type.value.upper()}] "
            
            # Add key information as context if it doesn't make content too long
            if key_info:
                key_context = f" (Key: {', '.join(key_info[:3])})"
                enhanced = type_prefix + content + key_context
            else:
                enhanced = type_prefix + content
            
            # Check if enhancement fits within optimal limit
            if self._estimate_tokens(enhanced) <= self.optimal_token_limit:
                return enhanced
            else:
                # Return original with just type prefix
                return type_prefix + content
                
        except Exception as e:
            logger.warning(f"Content enhancement failed: {e}")
            return content
    
    # ================================================================
    # QUALITY SCORING METHODS
    # ================================================================
    
    def _calculate_summary_quality_score(self, analysis: Dict[str, Any], entry_count: int) -> float:
        """Calculate quality score for session summary"""
        try:
            score = 0.0
            
            # Base score from entry count
            score += min(0.3, entry_count / 50.0)  # Max 0.3 for 50+ entries
            
            # Score from operation diversity
            operation_types = len(analysis.get("operation_types", {}))
            score += min(0.2, operation_types / 10.0)  # Max 0.2 for 10+ operation types
            
            # Score from key information extraction
            key_ops = len(analysis.get("key_operations", []))
            decisions = len(analysis.get("major_decisions", []))
            outcomes = len(analysis.get("outcomes_discoveries", []))
            
            score += min(0.2, key_ops / 5.0)      # Max 0.2 for 5+ key operations
            score += min(0.15, decisions / 3.0)   # Max 0.15 for 3+ decisions
            score += min(0.15, outcomes / 3.0)    # Max 0.15 for 3+ outcomes
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5  # Default moderate score
    
    def _calculate_embedding_quality_score(self, content: str, embedding_vector: List[float]) -> float:
        """Calculate quality score for generated embedding"""
        try:
            score = 0.0
            
            # Content length score
            content_length = len(content)
            if 50 <= content_length <= 300:  # Optimal range
                score += 0.3
            elif content_length > 10:
                score += 0.1
            
            # Embedding dimension score
            if len(embedding_vector) >= 384:  # Standard dimension
                score += 0.2
            
            # Embedding variance score (higher variance = more informative)
            if embedding_vector:
                variance = sum((x - sum(embedding_vector)/len(embedding_vector))**2 for x in embedding_vector) / len(embedding_vector)
                score += min(0.3, variance * 1000)  # Scale variance appropriately
            
            # Content complexity score
            word_count = len(content.split())
            if word_count >= 10:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Embedding quality score calculation failed: {e}")
            return 0.5  # Default moderate score
    
    # ================================================================
    # UTILITY AND HELPER METHODS
    # ================================================================
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get content processing statistics"""
        return {
            "optimal_token_limit": self.optimal_token_limit,
            "maximum_token_limit": self.maximum_token_limit,
            "token_estimation_ratio": self.token_estimation_ratio,
            "summary_limits": {
                "key_operations": self.summary_key_operations_limit,
                "decisions": self.summary_decisions_limit,
                "discoveries": self.summary_discoveries_limit
            },
            "entry_type_weights": {k.value: v for k, v in self.entry_type_weights.items()},
            "embedding_service_available": self.embedding_service.is_available() if self.embedding_service else False
        }
    
    def validate_content_for_processing(self, content: str) -> Dict[str, Any]:
        """Validate content for processing pipeline"""
        return {
            "content_length": len(content),
            "estimated_tokens": self._estimate_tokens(content),
            "needs_truncation": self._estimate_tokens(content) > self.optimal_token_limit,
            "exceeds_maximum": self._estimate_tokens(content) > self.maximum_token_limit,
            "is_empty": not content.strip(),
            "recommended_strategy": self._recommend_processing_strategy(content)
        }
    
    def _recommend_processing_strategy(self, content: str) -> str:
        """Recommend processing strategy for given content"""
        tokens = self._estimate_tokens(content)
        
        if tokens <= self.optimal_token_limit:
            return "direct_processing"
        elif tokens <= self.maximum_token_limit:
            return "smart_truncation"
        else:
            return "aggressive_summarization"