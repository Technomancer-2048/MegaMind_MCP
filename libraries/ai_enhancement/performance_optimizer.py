"""
Performance Optimization System
Optimizes embedding generation and retrieval performance
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import time
import numpy as np
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of optimization"""
    BATCH_SIZE = "batch_size"
    CACHE_STRATEGY = "cache_strategy"
    MODEL_SELECTION = "model_selection"
    PREPROCESSING = "preprocessing"
    INDEXING = "indexing"
    
class PerformanceMetric(Enum):
    """Performance metrics tracked"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUALITY_SCORE = "quality_score"

@dataclass
class PerformanceProfile:
    """Performance characteristics of a configuration"""
    configuration: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class OptimizationResult:
    """Result of optimization process"""
    optimization_type: OptimizationType
    original_config: Dict[str, Any]
    optimized_config: Dict[str, Any]
    improvement: Dict[str, float]
    confidence: float
    
@dataclass
class CacheEntry:
    """Entry in embedding cache"""
    chunk_id: str
    embedding: List[float]
    model: str
    created: datetime
    last_accessed: datetime
    access_count: int = 0

class PerformanceOptimizer:
    """System for optimizing embedding and retrieval performance"""
    
    def __init__(self, cache_size: int = 10000):
        self.cache_size = cache_size
        self.embedding_cache: Dict[str, CacheEntry] = {}
        self.performance_history: List[PerformanceProfile] = []
        self.optimization_history: List[OptimizationResult] = []
        self.metrics_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.current_config = self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'batch_size': 32,
            'model': 'all-MiniLM-L6-v2',
            'cache_strategy': 'lru',
            'preprocessing': {
                'cleaning_level': 'standard',
                'strip_formatting': True,
                'normalize_whitespace': True
            },
            'indexing': {
                'type': 'faiss',
                'metric': 'cosine',
                'nprobe': 10
            }
        }
        
    def optimize_batch_size(self, workload_characteristics: Dict[str, Any]) -> OptimizationResult:
        """Optimize batch size based on workload"""
        current_batch_size = self.current_config['batch_size']
        
        # Analyze workload
        avg_chunk_size = workload_characteristics.get('avg_chunk_size', 512)
        memory_available = workload_characteristics.get('memory_available_mb', 1024)
        target_latency = workload_characteristics.get('target_latency_ms', 100)
        
        # Calculate optimal batch size
        # Memory constraint
        max_batch_memory = int(memory_available * 0.7 / (avg_chunk_size * 0.001))
        
        # Latency constraint (empirical formula)
        max_batch_latency = int(target_latency / (avg_chunk_size * 0.0001))
        
        # Take minimum to satisfy both constraints
        optimal_batch_size = min(max_batch_memory, max_batch_latency, 128)
        optimal_batch_size = max(optimal_batch_size, 8)  # Minimum batch size
        
        # Round to power of 2 for efficiency
        optimal_batch_size = 2 ** int(np.log2(optimal_batch_size))
        
        improvement = {
            'throughput': (optimal_batch_size / current_batch_size - 1) * 100,
            'memory_efficiency': 15.0  # Estimated
        }
        
        return OptimizationResult(
            optimization_type=OptimizationType.BATCH_SIZE,
            original_config={'batch_size': current_batch_size},
            optimized_config={'batch_size': optimal_batch_size},
            improvement=improvement,
            confidence=0.85
        )
        
    def optimize_cache_strategy(self, access_patterns: Dict[str, Any]) -> OptimizationResult:
        """Optimize caching strategy based on access patterns"""
        current_strategy = self.current_config['cache_strategy']
        
        # Analyze access patterns
        hot_chunk_ratio = access_patterns.get('hot_chunk_ratio', 0.2)
        temporal_locality = access_patterns.get('temporal_locality', 0.5)
        access_frequency_skew = access_patterns.get('frequency_skew', 0.7)
        
        # Determine optimal strategy
        if hot_chunk_ratio > 0.3 and access_frequency_skew > 0.8:
            optimal_strategy = 'frequency_based'
        elif temporal_locality > 0.7:
            optimal_strategy = 'lru'
        elif hot_chunk_ratio < 0.1:
            optimal_strategy = 'random_replacement'
        else:
            optimal_strategy = 'adaptive_replacement'
            
        # Calculate expected improvement
        strategy_performance = {
            'lru': 0.7,
            'frequency_based': 0.8,
            'random_replacement': 0.5,
            'adaptive_replacement': 0.85
        }
        
        current_perf = strategy_performance.get(current_strategy, 0.6)
        optimal_perf = strategy_performance.get(optimal_strategy, 0.7)
        
        improvement = {
            'cache_hit_rate': (optimal_perf / current_perf - 1) * 100,
            'memory_efficiency': 10.0
        }
        
        return OptimizationResult(
            optimization_type=OptimizationType.CACHE_STRATEGY,
            original_config={'cache_strategy': current_strategy},
            optimized_config={'cache_strategy': optimal_strategy},
            improvement=improvement,
            confidence=0.75
        )
        
    def optimize_model_selection(self, requirements: Dict[str, Any]) -> OptimizationResult:
        """Select optimal embedding model based on requirements"""
        current_model = self.current_config['model']
        
        # Model characteristics
        models = {
            'all-MiniLM-L6-v2': {
                'speed': 100,
                'quality': 0.75,
                'memory': 100,
                'dimension': 384
            },
            'all-mpnet-base-v2': {
                'speed': 30,
                'quality': 0.90,
                'memory': 400,
                'dimension': 768
            },
            'all-MiniLM-L12-v2': {
                'speed': 50,
                'quality': 0.82,
                'memory': 200,
                'dimension': 384
            },
            'paraphrase-MiniLM-L6-v2': {
                'speed': 90,
                'quality': 0.78,
                'memory': 110,
                'dimension': 384
            }
        }
        
        # Requirements
        min_quality = requirements.get('min_quality', 0.7)
        max_latency = requirements.get('max_latency_ms', 100)
        max_memory = requirements.get('max_memory_mb', 500)
        
        # Score models
        model_scores = {}
        for model_name, specs in models.items():
            if specs['quality'] >= min_quality and specs['memory'] <= max_memory:
                # Composite score: balance speed and quality
                score = (specs['speed'] / max_latency) * 0.4 + specs['quality'] * 0.6
                model_scores[model_name] = score
                
        # Select best model
        if model_scores:
            optimal_model = max(model_scores.items(), key=lambda x: x[1])[0]
        else:
            optimal_model = current_model
            
        # Calculate improvement
        current_specs = models.get(current_model, models['all-MiniLM-L6-v2'])
        optimal_specs = models.get(optimal_model, current_specs)
        
        improvement = {
            'speed': (optimal_specs['speed'] / current_specs['speed'] - 1) * 100,
            'quality': (optimal_specs['quality'] / current_specs['quality'] - 1) * 100
        }
        
        return OptimizationResult(
            optimization_type=OptimizationType.MODEL_SELECTION,
            original_config={'model': current_model},
            optimized_config={'model': optimal_model},
            improvement=improvement,
            confidence=0.9
        )
        
    def optimize_preprocessing(self, content_analysis: Dict[str, Any]) -> OptimizationResult:
        """Optimize preprocessing based on content characteristics"""
        current_preprocessing = self.current_config['preprocessing'].copy()
        
        # Analyze content
        has_formatting = content_analysis.get('has_formatting', True)
        avg_whitespace_ratio = content_analysis.get('whitespace_ratio', 0.15)
        special_chars_ratio = content_analysis.get('special_chars_ratio', 0.05)
        
        # Optimize settings
        optimal_preprocessing = current_preprocessing.copy()
        
        if not has_formatting:
            optimal_preprocessing['strip_formatting'] = False
            
        if avg_whitespace_ratio > 0.2:
            optimal_preprocessing['normalize_whitespace'] = True
            optimal_preprocessing['aggressive_whitespace'] = True
            
        if special_chars_ratio > 0.1:
            optimal_preprocessing['cleaning_level'] = 'aggressive'
        elif special_chars_ratio < 0.02:
            optimal_preprocessing['cleaning_level'] = 'minimal'
            
        # Estimate improvement
        preprocessing_overhead = {
            'minimal': 1.0,
            'standard': 1.5,
            'aggressive': 2.0
        }
        
        current_overhead = preprocessing_overhead.get(
            current_preprocessing['cleaning_level'], 1.5
        )
        optimal_overhead = preprocessing_overhead.get(
            optimal_preprocessing['cleaning_level'], 1.5
        )
        
        improvement = {
            'preprocessing_speed': (current_overhead / optimal_overhead - 1) * 100,
            'embedding_quality': 5.0  # Estimated quality improvement
        }
        
        return OptimizationResult(
            optimization_type=OptimizationType.PREPROCESSING,
            original_config={'preprocessing': current_preprocessing},
            optimized_config={'preprocessing': optimal_preprocessing},
            improvement=improvement,
            confidence=0.7
        )
        
    def track_performance(self, operation: str, duration: float, 
                         config: Optional[Dict[str, Any]] = None):
        """Track performance metrics"""
        if config is None:
            config = self.current_config
            
        # Update metrics buffer
        self.metrics_buffer[f'{operation}_latency'].append(duration)
        
        # Calculate throughput
        if operation == 'batch_embedding':
            batch_size = config.get('batch_size', 32)
            throughput = batch_size / duration if duration > 0 else 0
            self.metrics_buffer['throughput'].append(throughput)
            
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights about current performance"""
        insights = {
            'current_config': self.current_config,
            'average_metrics': {},
            'trends': {},
            'bottlenecks': [],
            'optimization_opportunities': []
        }
        
        # Calculate averages
        for metric_name, values in self.metrics_buffer.items():
            if values:
                insights['average_metrics'][metric_name] = np.mean(values)
                
                # Calculate trend
                if len(values) > 10:
                    recent = np.mean(list(values)[-10:])
                    older = np.mean(list(values)[:10])
                    trend = (recent - older) / older if older > 0 else 0
                    insights['trends'][metric_name] = {
                        'direction': 'improving' if trend < 0 else 'degrading',
                        'change': abs(trend) * 100
                    }
                    
        # Identify bottlenecks
        avg_latency = insights['average_metrics'].get('embedding_latency', 0)
        if avg_latency > 100:
            insights['bottlenecks'].append({
                'type': 'high_latency',
                'severity': 'high' if avg_latency > 200 else 'medium',
                'metric': 'embedding_latency',
                'value': avg_latency
            })
            
        # Suggest optimizations
        cache_hit_rate = self._calculate_cache_hit_rate()
        if cache_hit_rate < 0.5:
            insights['optimization_opportunities'].append({
                'type': 'improve_caching',
                'potential_improvement': '20-30% latency reduction',
                'confidence': 0.8
            })
            
        return insights
        
    def apply_optimization(self, optimization: OptimizationResult):
        """Apply optimization result to current configuration"""
        if optimization.confidence > 0.6:  # Confidence threshold
            # Update configuration
            if optimization.optimization_type == OptimizationType.BATCH_SIZE:
                self.current_config['batch_size'] = optimization.optimized_config['batch_size']
                
            elif optimization.optimization_type == OptimizationType.CACHE_STRATEGY:
                self.current_config['cache_strategy'] = optimization.optimized_config['cache_strategy']
                
            elif optimization.optimization_type == OptimizationType.MODEL_SELECTION:
                self.current_config['model'] = optimization.optimized_config['model']
                
            elif optimization.optimization_type == OptimizationType.PREPROCESSING:
                self.current_config['preprocessing'] = optimization.optimized_config['preprocessing']
                
            # Record optimization
            self.optimization_history.append(optimization)
            logger.info(f"Applied {optimization.optimization_type.value} optimization")
            
    def manage_cache(self, chunk_id: str, embedding: Optional[List[float]] = None,
                    model: Optional[str] = None) -> Optional[List[float]]:
        """Manage embedding cache with configured strategy"""
        # Check cache hit
        if chunk_id in self.embedding_cache:
            entry = self.embedding_cache[chunk_id]
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            return entry.embedding
            
        # Cache miss - add if embedding provided
        if embedding and model:
            # Apply cache eviction if needed
            if len(self.embedding_cache) >= self.cache_size:
                self._evict_cache_entry()
                
            self.embedding_cache[chunk_id] = CacheEntry(
                chunk_id=chunk_id,
                embedding=embedding,
                model=model,
                created=datetime.now(),
                last_accessed=datetime.now(),
                access_count=1
            )
            
        return None
        
    def _evict_cache_entry(self):
        """Evict entry based on cache strategy"""
        if not self.embedding_cache:
            return
            
        strategy = self.current_config['cache_strategy']
        
        if strategy == 'lru':
            # Evict least recently used
            lru_entry = min(
                self.embedding_cache.items(),
                key=lambda x: x[1].last_accessed
            )
            del self.embedding_cache[lru_entry[0]]
            
        elif strategy == 'frequency_based':
            # Evict least frequently used
            lfu_entry = min(
                self.embedding_cache.items(),
                key=lambda x: x[1].access_count
            )
            del self.embedding_cache[lfu_entry[0]]
            
        elif strategy == 'random_replacement':
            # Random eviction
            import random
            random_key = random.choice(list(self.embedding_cache.keys()))
            del self.embedding_cache[random_key]
            
        elif strategy == 'adaptive_replacement':
            # Adaptive replacement cache (simplified)
            # Combine recency and frequency
            scores = {}
            now = datetime.now()
            for key, entry in self.embedding_cache.items():
                age = (now - entry.last_accessed).total_seconds()
                score = entry.access_count / (1 + age / 3600)  # Decay over hours
                scores[key] = score
                
            # Evict lowest score
            min_key = min(scores.items(), key=lambda x: x[1])[0]
            del self.embedding_cache[min_key]
            
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate current cache hit rate"""
        if not self.embedding_cache:
            return 0.0
            
        total_accesses = sum(entry.access_count for entry in self.embedding_cache.values())
        cache_hits = len(self.embedding_cache)
        
        return cache_hits / total_accesses if total_accesses > 0 else 0.0
        
    def generate_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        return {
            'current_performance': self.get_performance_insights(),
            'optimization_history': [
                {
                    'type': opt.optimization_type.value,
                    'improvement': opt.improvement,
                    'confidence': opt.confidence,
                    'timestamp': opt.optimized_config
                }
                for opt in self.optimization_history[-10:]  # Last 10 optimizations
            ],
            'cache_statistics': {
                'size': len(self.embedding_cache),
                'hit_rate': self._calculate_cache_hit_rate(),
                'strategy': self.current_config['cache_strategy']
            },
            'recommendations': self._generate_recommendations()
        }
        
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Check batch size optimization potential
        avg_latency = np.mean(self.metrics_buffer.get('embedding_latency', [0]))
        if avg_latency > 150:
            recommendations.append({
                'action': 'Increase batch size',
                'reason': 'High latency detected',
                'expected_improvement': '15-25% throughput increase'
            })
            
        # Check cache performance
        if self._calculate_cache_hit_rate() < 0.6:
            recommendations.append({
                'action': 'Optimize cache strategy',
                'reason': 'Low cache hit rate',
                'expected_improvement': '20-30% latency reduction'
            })
            
        return recommendations