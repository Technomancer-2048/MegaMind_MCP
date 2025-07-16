"""
Adaptive Learning Engine
Learns from user interactions to optimize chunking strategies and boundary detection
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of user feedback"""
    CHUNK_QUALITY = "chunk_quality"
    BOUNDARY_ACCURACY = "boundary_accuracy"
    RETRIEVAL_SUCCESS = "retrieval_success"
    MANUAL_CORRECTION = "manual_correction"
    
class LearningMetric(Enum):
    """Metrics tracked for learning"""
    CHUNK_SIZE_PREFERENCE = "chunk_size_preference"
    BOUNDARY_PATTERN = "boundary_pattern"
    QUALITY_THRESHOLD = "quality_threshold"
    RETRIEVAL_EFFECTIVENESS = "retrieval_effectiveness"

@dataclass
class UserFeedback:
    """Represents user feedback on system performance"""
    feedback_type: FeedbackType
    target_id: str  # chunk_id or document_id
    rating: float  # 0.0 to 1.0
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class LearningPattern:
    """Represents a learned pattern"""
    pattern_type: str
    pattern_data: Dict[str, Any]
    confidence: float
    occurrence_count: int
    last_seen: datetime
    success_rate: float

@dataclass  
class ChunkingStrategy:
    """Adaptive chunking strategy based on learned patterns"""
    name: str
    preferred_size: int
    boundary_patterns: List[str]
    quality_weights: Dict[str, float]
    confidence: float
    
@dataclass
class LearningState:
    """Current state of the learning system"""
    total_feedback_count: int
    patterns: List[LearningPattern]
    current_strategy: ChunkingStrategy
    performance_metrics: Dict[str, float]
    last_updated: datetime

class AdaptiveLearningEngine:
    """Engine that learns and adapts from user interactions"""
    
    def __init__(self):
        self.feedback_history: List[UserFeedback] = []
        self.learned_patterns: Dict[str, List[LearningPattern]] = defaultdict(list)
        self.performance_history: Dict[str, List[float]] = defaultdict(list)
        self.current_strategies: Dict[str, ChunkingStrategy] = {}
        self.learning_rate = 0.1
        self.min_feedback_threshold = 10
        self._initialize_default_strategies()
        
    def _initialize_default_strategies(self):
        """Initialize with default strategies"""
        self.current_strategies['default'] = ChunkingStrategy(
            name='default',
            preferred_size=512,
            boundary_patterns=[r'\n\n', r'\n#{1,6}\s', r'^\s*[-*]\s'],
            quality_weights={
                'readability': 0.15,
                'technical_accuracy': 0.25,
                'completeness': 0.20,
                'relevance': 0.15,
                'freshness': 0.10,
                'coherence': 0.10,
                'uniqueness': 0.03,
                'authority': 0.02
            },
            confidence=0.5
        )
        
    def record_feedback(self, feedback: UserFeedback):
        """Record user feedback for learning"""
        self.feedback_history.append(feedback)
        self.performance_history[feedback.feedback_type.value].append(feedback.rating)
        
        # Trigger learning if enough feedback accumulated
        if len(self.feedback_history) % self.min_feedback_threshold == 0:
            self._update_learning_patterns()
            
    def learn_optimal_chunk_size(self, feedback_data: List[UserFeedback]) -> Tuple[int, float]:
        """Learn optimal chunk size from feedback"""
        size_ratings = defaultdict(list)
        
        for feedback in feedback_data:
            if feedback.feedback_type == FeedbackType.CHUNK_QUALITY:
                chunk_size = feedback.details.get('chunk_size', 0)
                if chunk_size > 0:
                    size_ratings[chunk_size].append(feedback.rating)
                    
        if not size_ratings:
            return self.current_strategies['default'].preferred_size, 0.5
            
        # Calculate average rating for each size
        size_scores = {}
        for size, ratings in size_ratings.items():
            size_scores[size] = np.mean(ratings)
            
        # Find optimal size (weighted by frequency and score)
        optimal_size = max(size_scores.items(), key=lambda x: x[1])[0]
        confidence = size_scores[optimal_size]
        
        return optimal_size, confidence
        
    def learn_boundary_patterns(self, feedback_data: List[UserFeedback]) -> List[LearningPattern]:
        """Learn effective boundary patterns from feedback"""
        boundary_feedback = [
            f for f in feedback_data 
            if f.feedback_type == FeedbackType.BOUNDARY_ACCURACY
        ]
        
        pattern_performance = defaultdict(lambda: {'success': 0, 'total': 0})
        
        for feedback in boundary_feedback:
            patterns_used = feedback.details.get('boundary_patterns', [])
            success = feedback.rating > 0.7
            
            for pattern in patterns_used:
                pattern_performance[pattern]['total'] += 1
                if success:
                    pattern_performance[pattern]['success'] += 1
                    
        # Create learning patterns
        learned_patterns = []
        for pattern, stats in pattern_performance.items():
            if stats['total'] >= 5:  # Minimum occurrences
                success_rate = stats['success'] / stats['total']
                learned_patterns.append(LearningPattern(
                    pattern_type='boundary',
                    pattern_data={'regex': pattern},
                    confidence=min(stats['total'] / 20, 1.0),  # Confidence based on sample size
                    occurrence_count=stats['total'],
                    last_seen=datetime.now(),
                    success_rate=success_rate
                ))
                
        return sorted(learned_patterns, key=lambda x: x.success_rate, reverse=True)
        
    def adapt_quality_weights(self, feedback_data: List[UserFeedback]) -> Dict[str, float]:
        """Adapt quality dimension weights based on feedback"""
        quality_feedback = [
            f for f in feedback_data
            if f.feedback_type == FeedbackType.RETRIEVAL_SUCCESS
        ]
        
        if not quality_feedback:
            return self.current_strategies['default'].quality_weights
            
        # Track which dimensions correlate with successful retrievals
        dimension_impact = defaultdict(lambda: {'positive': 0, 'total': 0})
        
        for feedback in quality_feedback:
            quality_scores = feedback.details.get('quality_scores', {})
            success = feedback.rating > 0.7
            
            for dimension, score in quality_scores.items():
                dimension_impact[dimension]['total'] += 1
                if success and score > 0.7:
                    dimension_impact[dimension]['positive'] += 1
                    
        # Calculate new weights
        new_weights = {}
        total_impact = 0
        
        for dimension, impact in dimension_impact.items():
            if impact['total'] > 0:
                effectiveness = impact['positive'] / impact['total']
                new_weights[dimension] = effectiveness
                total_impact += effectiveness
                
        # Normalize weights
        if total_impact > 0:
            for dimension in new_weights:
                new_weights[dimension] /= total_impact
                
        # Blend with existing weights
        current_weights = self.current_strategies['default'].quality_weights
        blended_weights = {}
        
        for dimension in current_weights:
            old_weight = current_weights[dimension]
            new_weight = new_weights.get(dimension, old_weight)
            blended_weights[dimension] = (
                old_weight * (1 - self.learning_rate) + 
                new_weight * self.learning_rate
            )
            
        return blended_weights
        
    def predict_chunk_boundaries(self, content: str, context: Dict[str, Any]) -> List[int]:
        """Predict optimal chunk boundaries using learned patterns"""
        boundaries = []
        lines = content.split('\n')
        
        # Get current best patterns
        boundary_patterns = self.learned_patterns.get('boundary', [])
        if not boundary_patterns:
            # Use default patterns
            boundary_patterns = [
                LearningPattern(
                    pattern_type='boundary',
                    pattern_data={'regex': p},
                    confidence=0.5,
                    occurrence_count=0,
                    last_seen=datetime.now(),
                    success_rate=0.5
                )
                for p in self.current_strategies['default'].boundary_patterns
            ]
            
        # Apply patterns with confidence weighting
        for i, line in enumerate(lines):
            for pattern in boundary_patterns:
                import re
                if re.match(pattern.pattern_data['regex'], line):
                    # Weight by pattern confidence and success rate
                    score = pattern.confidence * pattern.success_rate
                    if score > 0.3:  # Threshold for boundary
                        boundaries.append(i)
                        break
                        
        return boundaries
        
    def optimize_embedding_strategy(self, usage_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize embedding generation based on usage patterns"""
        strategy = {
            'batch_size': 32,
            'model_preference': 'all-MiniLM-L6-v2',
            'cleaning_level': 'standard',
            'cache_strategy': 'lru'
        }
        
        # Analyze usage patterns
        avg_chunk_size = usage_patterns.get('avg_chunk_size', 512)
        retrieval_frequency = usage_patterns.get('retrieval_frequency', 'medium')
        quality_requirements = usage_patterns.get('quality_requirements', 'standard')
        
        # Adapt batch size based on chunk size
        if avg_chunk_size > 1000:
            strategy['batch_size'] = 16
        elif avg_chunk_size < 200:
            strategy['batch_size'] = 64
            
        # Adapt model based on quality requirements
        if quality_requirements == 'high':
            strategy['model_preference'] = 'all-mpnet-base-v2'
        elif quality_requirements == 'fast':
            strategy['model_preference'] = 'all-MiniLM-L6-v2'
            
        # Adapt caching based on retrieval frequency
        if retrieval_frequency == 'high':
            strategy['cache_strategy'] = 'preload'
        elif retrieval_frequency == 'low':
            strategy['cache_strategy'] = 'lazy'
            
        return strategy
        
    def _update_learning_patterns(self):
        """Update learning patterns based on accumulated feedback"""
        recent_feedback = self.feedback_history[-100:]  # Last 100 feedback items
        
        # Learn optimal chunk size
        optimal_size, size_confidence = self.learn_optimal_chunk_size(recent_feedback)
        
        # Learn boundary patterns
        boundary_patterns = self.learn_boundary_patterns(recent_feedback)
        self.learned_patterns['boundary'] = boundary_patterns[:10]  # Keep top 10
        
        # Adapt quality weights
        new_weights = self.adapt_quality_weights(recent_feedback)
        
        # Update strategy
        self.current_strategies['adaptive'] = ChunkingStrategy(
            name='adaptive',
            preferred_size=optimal_size,
            boundary_patterns=[p.pattern_data['regex'] for p in boundary_patterns[:5]],
            quality_weights=new_weights,
            confidence=size_confidence
        )
        
        logger.info(f"Updated adaptive strategy with {len(recent_feedback)} feedback items")
        
    def get_current_strategy(self, context: Optional[Dict[str, Any]] = None) -> ChunkingStrategy:
        """Get the current best strategy based on context"""
        if 'adaptive' in self.current_strategies:
            adaptive = self.current_strategies['adaptive']
            if adaptive.confidence > 0.7:
                return adaptive
                
        return self.current_strategies['default']
        
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about learned patterns and performance"""
        insights = {
            'total_feedback': len(self.feedback_history),
            'feedback_types': defaultdict(int),
            'average_ratings': {},
            'learned_patterns': {
                'boundary': len(self.learned_patterns.get('boundary', [])),
                'quality': len(self.learned_patterns.get('quality', []))
            },
            'current_strategy': self.get_current_strategy().name,
            'performance_trend': {}
        }
        
        # Analyze feedback distribution
        for feedback in self.feedback_history:
            insights['feedback_types'][feedback.feedback_type.value] += 1
            
        # Calculate average ratings
        for metric, ratings in self.performance_history.items():
            if ratings:
                insights['average_ratings'][metric] = np.mean(ratings)
                
        # Calculate performance trends (last 30 days)
        cutoff = datetime.now() - timedelta(days=30)
        recent_feedback = [f for f in self.feedback_history if f.timestamp > cutoff]
        
        if recent_feedback:
            recent_avg = np.mean([f.rating for f in recent_feedback])
            overall_avg = np.mean([f.rating for f in self.feedback_history])
            insights['performance_trend']['direction'] = 'improving' if recent_avg > overall_avg else 'declining'
            insights['performance_trend']['change'] = recent_avg - overall_avg
            
        return insights
        
    def export_learning_state(self) -> LearningState:
        """Export current learning state for persistence"""
        return LearningState(
            total_feedback_count=len(self.feedback_history),
            patterns=list(self.learned_patterns.get('boundary', [])),
            current_strategy=self.get_current_strategy(),
            performance_metrics={
                metric: np.mean(ratings) if ratings else 0.0
                for metric, ratings in self.performance_history.items()
            },
            last_updated=datetime.now()
        )
        
    def import_learning_state(self, state: LearningState):
        """Import learning state from persistence"""
        # Restore patterns
        self.learned_patterns['boundary'] = state.patterns
        
        # Restore strategy
        self.current_strategies[state.current_strategy.name] = state.current_strategy
        
        # Note: Full feedback history would need separate storage
        logger.info(f"Imported learning state with {state.total_feedback_count} feedback items")