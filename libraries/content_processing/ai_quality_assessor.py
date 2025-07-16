#!/usr/bin/env python3
"""
AI Quality Assessor for Enhanced Multi-Embedding Entry System
Adapted from Phase 8 ai_powered_curator.py for quality assessment
"""

import logging
import statistics
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .intelligent_chunker import Chunk, ChunkType

logger = logging.getLogger(__name__)

class QualityDimension(Enum):
    """Quality assessment dimensions"""
    READABILITY = "readability"
    TECHNICAL_ACCURACY = "technical_accuracy"
    COMPLETENESS = "completeness"
    RELEVANCE = "relevance"
    FRESHNESS = "freshness"
    COHERENCE = "coherence"
    UNIQUENESS = "uniqueness"
    AUTHORITY = "authority"

@dataclass
class QualityScore:
    """Multi-dimensional quality score"""
    chunk_id: str
    overall_score: float
    dimension_scores: Dict[QualityDimension, float]
    confidence: float
    assessment_metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class QualityIssue:
    """Identified quality issue"""
    dimension: QualityDimension
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    suggested_fix: Optional[str] = None
    impact_score: float = 0.0

class AIQualityAssessor:
    """
    AI-powered quality assessment for chunks
    Adapted from Phase 8 curation engine
    """
    
    def __init__(self):
        # Quality dimension weights (must sum to 1.0)
        self.dimension_weights = {
            QualityDimension.READABILITY: 0.15,
            QualityDimension.TECHNICAL_ACCURACY: 0.25,
            QualityDimension.COMPLETENESS: 0.20,
            QualityDimension.RELEVANCE: 0.15,
            QualityDimension.FRESHNESS: 0.10,
            QualityDimension.COHERENCE: 0.10,
            QualityDimension.UNIQUENESS: 0.03,
            QualityDimension.AUTHORITY: 0.02
        }
        
        # Thresholds for quality levels
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.7,
            'acceptable': 0.5,
            'poor': 0.3
        }
        
        # Readability metrics thresholds
        self.readability_thresholds = {
            'optimal_sentence_length': (10, 25),
            'optimal_word_length': (4, 8),
            'optimal_paragraph_length': (50, 200)
        }
        
        logger.info("AIQualityAssessor initialized with 8-dimensional assessment")
    
    def assess_chunk_quality(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> QualityScore:
        """
        Perform comprehensive quality assessment on a chunk
        
        Args:
            chunk: Chunk to assess
            context: Optional surrounding chunks for context
            
        Returns:
            QualityScore with multi-dimensional assessment
        """
        dimension_scores = {}
        
        # Assess each dimension
        dimension_scores[QualityDimension.READABILITY] = self._assess_readability(chunk)
        dimension_scores[QualityDimension.TECHNICAL_ACCURACY] = self._assess_technical_accuracy(chunk)
        dimension_scores[QualityDimension.COMPLETENESS] = self._assess_completeness(chunk)
        dimension_scores[QualityDimension.RELEVANCE] = self._assess_relevance(chunk, context)
        dimension_scores[QualityDimension.FRESHNESS] = self._assess_freshness(chunk)
        dimension_scores[QualityDimension.COHERENCE] = self._assess_coherence(chunk, context)
        dimension_scores[QualityDimension.UNIQUENESS] = self._assess_uniqueness(chunk, context)
        dimension_scores[QualityDimension.AUTHORITY] = self._assess_authority(chunk)
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_scores)
        
        # Calculate confidence
        confidence = self._calculate_confidence(dimension_scores, chunk)
        
        # Build assessment metadata
        metadata = {
            'chunk_type': chunk.chunk_type.value,
            'token_count': chunk.token_count,
            'quality_level': self._get_quality_level(overall_score),
            'issues_found': self._identify_quality_issues(dimension_scores),
            'improvement_suggestions': self._generate_improvement_suggestions(dimension_scores, chunk)
        }
        
        return QualityScore(
            chunk_id=chunk.chunk_id,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            confidence=confidence,
            assessment_metadata=metadata
        )
    
    def validate_technical_accuracy(self, content: str, chunk_type: ChunkType) -> float:
        """
        Validate technical accuracy of content
        
        Args:
            content: Content to validate
            chunk_type: Type of chunk
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        score = 0.5  # Base score
        
        # Check for common technical issues
        issues = []
        
        # Code-specific checks
        if chunk_type == ChunkType.CODE_BLOCK:
            # Check for syntax patterns (simplified)
            if self._check_code_syntax(content):
                score += 0.2
            else:
                issues.append("potential syntax issues")
            
            # Check for common antipatterns
            if not self._check_code_antipatterns(content):
                score += 0.2
            else:
                issues.append("code antipatterns detected")
        
        # Documentation checks
        else:
            # Check for technical terminology consistency
            if self._check_terminology_consistency(content):
                score += 0.2
            
            # Check for factual accuracy markers
            if self._check_factual_markers(content):
                score += 0.2
        
        # Apply penalties for issues
        score -= len(issues) * 0.1
        
        return max(0.0, min(1.0, score))
    
    def measure_coherence(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> float:
        """
        Measure semantic coherence of chunk
        
        Args:
            chunk: Chunk to measure
            context: Optional context chunks
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        coherence_score = chunk.semantic_coherence  # Start with pre-calculated score
        
        # Adjust based on content analysis
        content = chunk.content.lower()
        
        # Check for coherence indicators
        coherence_indicators = [
            # Logical connectors
            any(connector in content for connector in ['therefore', 'thus', 'hence', 'because', 'since']),
            # Sequential markers
            any(marker in content for marker in ['first', 'second', 'then', 'finally', 'next']),
            # Reference markers
            any(ref in content for ref in ['as mentioned', 'as discussed', 'previously', 'above', 'below']),
            # Consistent terminology
            self._check_terminology_consistency(content)
        ]
        
        # Calculate bonus
        coherence_bonus = sum(coherence_indicators) * 0.1
        coherence_score = min(1.0, coherence_score + coherence_bonus)
        
        # Context-based adjustment
        if context:
            context_similarity = self._calculate_context_similarity(chunk, context)
            coherence_score = (coherence_score * 0.7) + (context_similarity * 0.3)
        
        return coherence_score
    
    def evaluate_completeness(self, chunk: Chunk) -> float:
        """
        Evaluate content completeness
        
        Args:
            chunk: Chunk to evaluate
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        score = 0.5  # Base score
        
        # Check chunk type specific completeness
        if chunk.chunk_type == ChunkType.CODE_BLOCK:
            score = self._evaluate_code_completeness(chunk)
        elif chunk.chunk_type == ChunkType.LIST_SECTION:
            score = self._evaluate_list_completeness(chunk)
        elif chunk.chunk_type == ChunkType.PARAGRAPH:
            score = self._evaluate_paragraph_completeness(chunk)
        else:
            # Generic completeness check
            content_length = len(chunk.content.split())
            if content_length < 20:
                score = 0.3  # Too short
            elif content_length > 200:
                score = 0.9  # Comprehensive
            else:
                score = 0.5 + (content_length - 20) / 360  # Linear scale
        
        # Check for incomplete sentences
        if chunk.content.strip() and not chunk.content.strip()[-1] in '.!?':
            score *= 0.9  # Penalty for incomplete ending
        
        # Check for truncation indicators
        if any(indicator in chunk.content.lower() for indicator in ['...', 'etc', 'and so on']):
            score *= 0.95  # Slight penalty for truncation
        
        return min(1.0, score)
    
    def _assess_readability(self, chunk: Chunk) -> float:
        """Assess readability of chunk content"""
        content = chunk.content
        
        # Skip readability for code blocks
        if chunk.chunk_type == ChunkType.CODE_BLOCK:
            return 0.9  # Code readability is different
        
        # Calculate readability metrics
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        words = content.split()
        
        if not sentences or not words:
            return 0.5
        
        # Average sentence length
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        sentence_score = self._score_in_range(
            avg_sentence_length,
            self.readability_thresholds['optimal_sentence_length']
        )
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        word_score = self._score_in_range(
            avg_word_length,
            self.readability_thresholds['optimal_word_length']
        )
        
        # Paragraph length (for non-list chunks)
        if chunk.chunk_type != ChunkType.LIST_SECTION:
            paragraph_score = self._score_in_range(
                len(words),
                self.readability_thresholds['optimal_paragraph_length']
            )
        else:
            paragraph_score = 0.8  # Lists have different optimal lengths
        
        # Combine scores
        readability_score = (sentence_score * 0.4 + word_score * 0.3 + paragraph_score * 0.3)
        
        return readability_score
    
    def _assess_technical_accuracy(self, chunk: Chunk) -> float:
        """Assess technical accuracy"""
        return self.validate_technical_accuracy(chunk.content, chunk.chunk_type)
    
    def _assess_completeness(self, chunk: Chunk) -> float:
        """Assess completeness"""
        return self.evaluate_completeness(chunk)
    
    def _assess_relevance(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> float:
        """Assess relevance of chunk"""
        # Base relevance from chunk metadata
        base_relevance = chunk.metadata.get('relevance_score', 0.5)
        
        # Adjust based on chunk type
        type_bonus = {
            ChunkType.HEADING: 0.1,
            ChunkType.EXAMPLE: 0.15,
            ChunkType.DEFINITION: 0.15,
            ChunkType.RULE: 0.2,
            ChunkType.CODE_BLOCK: 0.1
        }
        
        relevance = base_relevance + type_bonus.get(chunk.chunk_type, 0.0)
        
        # Context-based adjustment
        if context:
            # Check if chunk relates to context
            context_keywords = self._extract_keywords(context)
            chunk_keywords = self._extract_keywords([chunk])
            overlap = len(set(chunk_keywords) & set(context_keywords))
            if overlap > 0:
                relevance += min(0.2, overlap * 0.05)
        
        return min(1.0, relevance)
    
    def _assess_freshness(self, chunk: Chunk) -> float:
        """Assess content freshness"""
        # Check for temporal indicators
        content_lower = chunk.content.lower()
        
        # Outdated indicators
        outdated_patterns = ['deprecated', 'legacy', 'obsolete', 'outdated', 'old version']
        if any(pattern in content_lower for pattern in outdated_patterns):
            return 0.3
        
        # Current indicators
        current_patterns = ['latest', 'current', 'new', 'updated', 'recent']
        if any(pattern in content_lower for pattern in current_patterns):
            return 0.9
        
        # Check metadata for dates
        if 'last_updated' in chunk.metadata:
            last_updated = chunk.metadata['last_updated']
            days_old = (datetime.now() - last_updated).days
            if days_old < 30:
                return 0.95
            elif days_old < 90:
                return 0.8
            elif days_old < 365:
                return 0.6
            else:
                return 0.4
        
        # Default freshness
        return 0.7
    
    def _assess_coherence(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> float:
        """Assess coherence"""
        return self.measure_coherence(chunk, context)
    
    def _assess_uniqueness(self, chunk: Chunk, context: Optional[List[Chunk]] = None) -> float:
        """Assess uniqueness of content"""
        if not context:
            return 0.8  # Default uniqueness
        
        # Simple uniqueness check based on content overlap
        chunk_words = set(chunk.content.lower().split())
        
        overlap_scores = []
        for other_chunk in context:
            if other_chunk.chunk_id != chunk.chunk_id:
                other_words = set(other_chunk.content.lower().split())
                overlap = len(chunk_words & other_words) / len(chunk_words) if chunk_words else 0
                overlap_scores.append(1 - overlap)
        
        if overlap_scores:
            return sum(overlap_scores) / len(overlap_scores)
        
        return 0.8
    
    def _assess_authority(self, chunk: Chunk) -> float:
        """Assess content authority/reliability"""
        content_lower = chunk.content.lower()
        
        # Authority indicators
        authority_score = 0.5  # Base score
        
        # Positive indicators
        if any(indicator in content_lower for indicator in ['research', 'study', 'paper', 'publication']):
            authority_score += 0.2
        
        if any(indicator in content_lower for indicator in ['official', 'standard', 'specification']):
            authority_score += 0.15
        
        # Citations or references
        if any(pattern in chunk.content for pattern in ['[1]', '[2]', 'et al.', 'Reference:']):
            authority_score += 0.15
        
        # Negative indicators
        if any(indicator in content_lower for indicator in ['maybe', 'perhaps', 'might be', 'possibly']):
            authority_score -= 0.1
        
        return max(0.0, min(1.0, authority_score))
    
    def _calculate_overall_score(self, dimension_scores: Dict[QualityDimension, float]) -> float:
        """Calculate weighted overall score"""
        overall = 0.0
        for dimension, score in dimension_scores.items():
            weight = self.dimension_weights.get(dimension, 0.0)
            overall += score * weight
        return overall
    
    def _calculate_confidence(self, dimension_scores: Dict[QualityDimension, float], chunk: Chunk) -> float:
        """Calculate confidence in assessment"""
        # Base confidence on score consistency
        scores = list(dimension_scores.values())
        if len(scores) > 1:
            std_dev = statistics.stdev(scores)
            # Lower standard deviation = higher confidence
            confidence = 1.0 - min(std_dev, 0.5)
        else:
            confidence = 0.5
        
        # Adjust based on chunk properties
        if chunk.token_count < 20:
            confidence *= 0.8  # Less confident on very short chunks
        elif chunk.token_count > 300:
            confidence *= 1.1  # More confident on substantial chunks
        
        return min(1.0, confidence)
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level from score"""
        for level, threshold in sorted(self.quality_thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return 'poor'
    
    def _identify_quality_issues(self, dimension_scores: Dict[QualityDimension, float]) -> List[QualityIssue]:
        """Identify quality issues from dimension scores"""
        issues = []
        
        for dimension, score in dimension_scores.items():
            if score < 0.5:  # Below acceptable
                severity = 'critical' if score < 0.3 else 'high'
                issue = QualityIssue(
                    dimension=dimension,
                    severity=severity,
                    description=f"Low {dimension.value} score: {score:.2f}",
                    impact_score=self.dimension_weights[dimension] * (0.5 - score)
                )
                
                # Add specific suggestions
                if dimension == QualityDimension.READABILITY:
                    issue.suggested_fix = "Simplify sentences and use clearer language"
                elif dimension == QualityDimension.COMPLETENESS:
                    issue.suggested_fix = "Add more details or examples"
                elif dimension == QualityDimension.COHERENCE:
                    issue.suggested_fix = "Improve logical flow and connections"
                
                issues.append(issue)
        
        return sorted(issues, key=lambda x: x.impact_score, reverse=True)
    
    def _generate_improvement_suggestions(self, 
                                        dimension_scores: Dict[QualityDimension, float],
                                        chunk: Chunk) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Readability improvements
        if dimension_scores[QualityDimension.READABILITY] < 0.7:
            suggestions.append("Break long sentences into shorter, clearer ones")
            if chunk.chunk_type == ChunkType.PARAGRAPH:
                suggestions.append("Consider using bullet points for lists")
        
        # Completeness improvements
        if dimension_scores[QualityDimension.COMPLETENESS] < 0.7:
            if chunk.chunk_type == ChunkType.CODE_BLOCK:
                suggestions.append("Add comments to explain complex code sections")
            else:
                suggestions.append("Provide more context or examples")
        
        # Coherence improvements
        if dimension_scores[QualityDimension.COHERENCE] < 0.7:
            suggestions.append("Add transition phrases between ideas")
            suggestions.append("Ensure consistent terminology throughout")
        
        # Technical accuracy improvements
        if dimension_scores[QualityDimension.TECHNICAL_ACCURACY] < 0.7:
            suggestions.append("Review technical details for accuracy")
            if chunk.chunk_type == ChunkType.CODE_BLOCK:
                suggestions.append("Test code for syntax and logic errors")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    # Helper methods
    
    def _score_in_range(self, value: float, optimal_range: Tuple[float, float]) -> float:
        """Score a value based on optimal range"""
        min_val, max_val = optimal_range
        if min_val <= value <= max_val:
            return 1.0
        elif value < min_val:
            return max(0.0, 1.0 - (min_val - value) / min_val)
        else:
            return max(0.0, 1.0 - (value - max_val) / max_val)
    
    def _check_code_syntax(self, content: str) -> bool:
        """Simple code syntax check"""
        # Basic bracket/parenthesis matching
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for char in content:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack:
                    return False
                if brackets[stack[-1]] != char:
                    return False
                stack.pop()
        return len(stack) == 0
    
    def _check_code_antipatterns(self, content: str) -> bool:
        """Check for common code antipatterns"""
        antipatterns = [
            'catch(Exception)',  # Too broad exception catching
            'SELECT *',  # SQL antipattern
            'eval(',  # Security risk
            'TODO',  # Incomplete code
            'FIXME',  # Known issues
        ]
        content_lower = content.lower()
        return any(pattern.lower() in content_lower for pattern in antipatterns)
    
    def _check_terminology_consistency(self, content: str) -> bool:
        """Check for consistent terminology usage"""
        # Simple check for now - could be enhanced with terminology dictionary
        words = content.lower().split()
        # Check for mixed terminology (simplified)
        mixed_terms = [
            ('function', 'method'),
            ('parameter', 'argument'),
            ('property', 'attribute'),
        ]
        
        for term1, term2 in mixed_terms:
            if term1 in words and term2 in words:
                return False
        return True
    
    def _check_factual_markers(self, content: str) -> bool:
        """Check for factual accuracy markers"""
        positive_markers = ['according to', 'research shows', 'studies indicate', 'documented in']
        negative_markers = ['might be', 'could be', 'possibly', 'uncertain']
        
        content_lower = content.lower()
        positive_count = sum(1 for marker in positive_markers if marker in content_lower)
        negative_count = sum(1 for marker in negative_markers if marker in content_lower)
        
        return positive_count > negative_count
    
    def _calculate_context_similarity(self, chunk: Chunk, context: List[Chunk]) -> float:
        """Calculate similarity between chunk and context"""
        if not context:
            return 0.5
        
        # Simple keyword-based similarity
        chunk_words = set(chunk.content.lower().split())
        
        similarities = []
        for context_chunk in context:
            context_words = set(context_chunk.content.lower().split())
            if chunk_words and context_words:
                similarity = len(chunk_words & context_words) / len(chunk_words | context_words)
                similarities.append(similarity)
        
        return sum(similarities) / len(similarities) if similarities else 0.5
    
    def _extract_keywords(self, chunks: List[Chunk]) -> List[str]:
        """Extract keywords from chunks"""
        # Simple keyword extraction - could be enhanced with TF-IDF
        all_words = []
        for chunk in chunks:
            words = chunk.content.lower().split()
            # Filter out common words (simplified stopword removal)
            filtered_words = [w for w in words if len(w) > 3 and w.isalnum()]
            all_words.extend(filtered_words)
        
        # Return most common words
        word_counts = {}
        for word in all_words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:20]]  # Top 20 keywords
    
    def _evaluate_code_completeness(self, chunk: Chunk) -> float:
        """Evaluate completeness of code chunk"""
        content = chunk.content
        score = 0.5
        
        # Check for function/class definitions
        if any(pattern in content for pattern in ['def ', 'class ', 'function ', 'const ', 'var ']):
            score += 0.2
        
        # Check for proper structure
        if self._check_code_syntax(content):
            score += 0.2
        
        # Check for imports/includes
        if any(pattern in content for pattern in ['import ', 'include ', 'require', 'using ']):
            score += 0.1
        
        return min(1.0, score)
    
    def _evaluate_list_completeness(self, chunk: Chunk) -> float:
        """Evaluate completeness of list chunk"""
        # Count list items
        list_items = len([line for line in chunk.content.split('\n') if line.strip().startswith(('•', '-', '*', '1', '2', '3'))])
        
        if list_items < 3:
            return 0.5  # Very short list
        elif list_items < 5:
            return 0.7
        elif list_items < 10:
            return 0.9
        else:
            return 1.0  # Comprehensive list
    
    def _evaluate_paragraph_completeness(self, chunk: Chunk) -> float:
        """Evaluate completeness of paragraph chunk"""
        sentences = [s.strip() for s in chunk.content.split('.') if s.strip()]
        
        if len(sentences) < 2:
            return 0.4  # Too short
        elif len(sentences) < 4:
            return 0.7
        elif len(sentences) < 8:
            return 0.9
        else:
            return 1.0  # Complete paragraph