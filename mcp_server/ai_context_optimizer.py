#!/usr/bin/env python3
"""
Phase 8: AI-Powered Context Optimization Engine
Advanced AI algorithms for intelligent context curation and delivery optimization
"""

import json
import logging
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
import uuid
import statistics

# AI and ML imports for advanced optimization
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Advanced ML libraries not available - using fallback AI optimization")

logger = logging.getLogger(__name__)

@dataclass
class ContextOptimizationRequest:
    """Context optimization request structure"""
    request_id: str
    query: str
    session_context: Dict[str, Any]
    model_type: str  # 'sonnet', 'opus', 'haiku'
    task_complexity: str  # 'simple', 'medium', 'complex'
    token_budget: int
    optimization_goals: List[str]  # 'accuracy', 'speed', 'diversity', 'coherence'
    timestamp: datetime

@dataclass
class OptimizedContext:
    """Optimized context delivery structure"""
    context_id: str
    request_id: str
    chunks: List[Dict[str, Any]]
    optimization_score: float
    relevance_scores: List[float]
    token_usage: int
    optimization_strategy: str
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class AIOptimizationMetrics:
    """AI optimization performance metrics"""
    optimization_id: str
    request_count: int
    avg_optimization_time_ms: float
    avg_relevance_score: float
    token_efficiency: float
    success_rate: float
    improvement_rate: float
    timestamp: datetime

class AdaptiveLearningEngine:
    """
    Adaptive learning system that improves optimization over time
    """
    
    def __init__(self):
        self.learning_history = deque(maxlen=10000)
        self.performance_models = {}
        self.feature_scalers = {}
        self.optimization_patterns = defaultdict(list)
        
        # Learning configuration
        self.learning_threshold = 0.1  # Minimum improvement to trigger model update
        self.model_retrain_interval = 1000  # Retrain after N optimizations
        self.feedback_weight = 0.3  # Weight of user feedback vs. automatic metrics
        
        logger.info("✅ Adaptive Learning Engine initialized")
    
    def record_optimization(self, request: ContextOptimizationRequest, 
                          result: OptimizedContext, feedback: Optional[Dict[str, Any]] = None):
        """Record optimization for learning purposes"""
        learning_record = {
            'request_features': self._extract_request_features(request),
            'optimization_strategy': result.optimization_strategy,
            'performance_metrics': {
                'relevance_score': result.optimization_score,
                'token_efficiency': result.token_usage / request.token_budget,
                'confidence': result.confidence
            },
            'user_feedback': feedback,
            'timestamp': datetime.now()
        }
        
        self.learning_history.append(learning_record)
        
        # Update optimization patterns
        pattern_key = f"{request.model_type}_{request.task_complexity}"
        self.optimization_patterns[pattern_key].append(learning_record)
        
        # Trigger model update if threshold reached
        if len(self.learning_history) % self.model_retrain_interval == 0:
            self._update_performance_models()
    
    def _extract_request_features(self, request: ContextOptimizationRequest) -> Dict[str, float]:
        """Extract features from optimization request"""
        return {
            'query_length': len(request.query.split()),
            'query_complexity': self._calculate_query_complexity(request.query),
            'session_depth': len(request.session_context.get('previous_queries', [])),
            'token_budget_ratio': request.token_budget / 100000,  # Normalize
            'goal_count': len(request.optimization_goals),
            'model_capability': self._get_model_capability_score(request.model_type),
            'task_complexity_score': self._get_complexity_score(request.task_complexity)
        }
    
    def _calculate_query_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        complexity_indicators = [
            len(query.split()) > 10,  # Long query
            '?' in query,  # Question
            any(word in query.lower() for word in ['implement', 'create', 'build', 'design']),  # Implementation
            any(word in query.lower() for word in ['why', 'how', 'what', 'when', 'where']),  # Investigation
            any(word in query.lower() for word in ['optimize', 'improve', 'enhance', 'refactor'])  # Optimization
        ]
        return sum(complexity_indicators) / len(complexity_indicators)
    
    def _get_model_capability_score(self, model_type: str) -> float:
        """Get normalized model capability score"""
        capabilities = {
            'haiku': 0.3,
            'sonnet': 0.7,
            'opus': 1.0
        }
        return capabilities.get(model_type.lower(), 0.5)
    
    def _get_complexity_score(self, task_complexity: str) -> float:
        """Get normalized task complexity score"""
        complexities = {
            'simple': 0.2,
            'medium': 0.5,
            'complex': 1.0
        }
        return complexities.get(task_complexity.lower(), 0.5)
    
    def predict_optimal_strategy(self, request: ContextOptimizationRequest) -> Dict[str, Any]:
        """Predict optimal optimization strategy for request"""
        features = self._extract_request_features(request)
        pattern_key = f"{request.model_type}_{request.task_complexity}"
        
        # Get historical patterns for similar requests
        similar_patterns = self.optimization_patterns.get(pattern_key, [])
        
        if len(similar_patterns) < 5:
            # Fall back to default strategy
            return self._get_default_strategy(request)
        
        # Analyze successful patterns
        successful_patterns = [
            p for p in similar_patterns[-100:]  # Recent patterns
            if p['performance_metrics']['relevance_score'] > 0.8
        ]
        
        if not successful_patterns:
            return self._get_default_strategy(request)
        
        # Find most common successful strategy
        strategy_counts = defaultdict(int)
        for pattern in successful_patterns:
            strategy_counts[pattern['optimization_strategy']] += 1
        
        optimal_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'strategy': optimal_strategy,
            'confidence': len(successful_patterns) / len(similar_patterns),
            'expected_performance': statistics.mean(
                p['performance_metrics']['relevance_score'] for p in successful_patterns
            )
        }
    
    def _get_default_strategy(self, request: ContextOptimizationRequest) -> Dict[str, Any]:
        """Get default optimization strategy"""
        if request.task_complexity == 'simple':
            strategy = 'fast_relevance'
        elif request.task_complexity == 'complex':
            strategy = 'comprehensive_analysis'
        else:
            strategy = 'balanced_optimization'
        
        return {
            'strategy': strategy,
            'confidence': 0.5,
            'expected_performance': 0.7
        }
    
    def _update_performance_models(self):
        """Update ML models based on learning history"""
        if not ML_AVAILABLE or len(self.learning_history) < 100:
            return
        
        try:
            # Prepare training data
            features = []
            targets = []
            
            for record in list(self.learning_history)[-1000:]:  # Use recent history
                features.append(list(record['request_features'].values()))
                targets.append(record['performance_metrics']['relevance_score'])
            
            X = np.array(features)
            y = np.array(targets)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingRegressor(random_state=42),
                'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
            }
            
            best_model = None
            best_score = -1
            
            for model_name, model in models.items():
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model = model_name
                    self.performance_models['relevance_predictor'] = model
                    self.feature_scalers['relevance_predictor'] = scaler
            
            logger.info(f"✅ Updated performance models - Best: {best_model} (Score: {best_score:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to update performance models: {e}")

class IntelligentContextAssembler:
    """
    Advanced context assembly with multi-modal fusion
    """
    
    def __init__(self, learning_engine: AdaptiveLearningEngine):
        self.learning_engine = learning_engine
        self.vectorizer = None
        self.chunk_vectors = {}
        self.context_cache = {}
        
        # Assembly strategies
        self.assembly_strategies = {
            'fast_relevance': self._fast_relevance_assembly,
            'comprehensive_analysis': self._comprehensive_analysis_assembly,
            'balanced_optimization': self._balanced_optimization_assembly,
            'semantic_clustering': self._semantic_clustering_assembly,
            'temporal_relevance': self._temporal_relevance_assembly
        }
        
        # Configuration
        self.max_cache_size = 1000
        self.similarity_threshold = 0.3
        self.diversity_threshold = 0.2
        
        logger.info("✅ Intelligent Context Assembler initialized")
    
    def assemble_context(self, request: ContextOptimizationRequest, 
                        available_chunks: List[Dict[str, Any]]) -> OptimizedContext:
        """Assemble optimized context based on request requirements"""
        start_time = time.time()
        
        # Get optimal strategy from learning engine
        strategy_prediction = self.learning_engine.predict_optimal_strategy(request)
        strategy = strategy_prediction['strategy']
        
        logger.info(f"🤖 Using AI strategy: {strategy} (confidence: {strategy_prediction['confidence']:.3f})")
        
        # Apply assembly strategy
        assembly_function = self.assembly_strategies.get(strategy, self._balanced_optimization_assembly)
        optimized_chunks = assembly_function(request, available_chunks)
        
        # Calculate optimization metrics
        optimization_score = self._calculate_optimization_score(request, optimized_chunks)
        relevance_scores = [chunk.get('relevance_score', 0.5) for chunk in optimized_chunks]
        token_usage = sum(chunk.get('token_count', 100) for chunk in optimized_chunks)
        
        # Create optimized context
        context = OptimizedContext(
            context_id=str(uuid.uuid4()),
            request_id=request.request_id,
            chunks=optimized_chunks,
            optimization_score=optimization_score,
            relevance_scores=relevance_scores,
            token_usage=token_usage,
            optimization_strategy=strategy,
            confidence=strategy_prediction['confidence'],
            metadata={
                'assembly_time_ms': (time.time() - start_time) * 1000,
                'strategy_prediction': strategy_prediction,
                'optimization_goals': request.optimization_goals
            },
            timestamp=datetime.now()
        )
        
        # Record for learning
        self.learning_engine.record_optimization(request, context)
        
        return context
    
    def _fast_relevance_assembly(self, request: ContextOptimizationRequest,
                                chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fast relevance-based assembly for simple tasks"""
        # Sort by basic relevance score
        sorted_chunks = sorted(
            chunks,
            key=lambda x: x.get('relevance_score', 0),
            reverse=True
        )
        
        # Select top chunks within token budget
        selected_chunks = []
        token_count = 0
        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.get('token_count', 100)
            if token_count + chunk_tokens <= request.token_budget:
                selected_chunks.append(chunk)
                token_count += chunk_tokens
            else:
                break
        
        return selected_chunks[:10]  # Limit for fast processing
    
    def _comprehensive_analysis_assembly(self, request: ContextOptimizationRequest,
                                       chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Comprehensive analysis for complex tasks"""
        if not ML_AVAILABLE:
            return self._balanced_optimization_assembly(request, chunks)
        
        try:
            # Extract text content
            chunk_texts = [chunk.get('content', '') for chunk in chunks]
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            chunk_vectors = vectorizer.fit_transform(chunk_texts)
            query_vector = vectorizer.transform([request.query])
            
            # Calculate semantic similarity
            similarities = cosine_similarity(query_vector, chunk_vectors)[0]
            
            # Calculate diversity scores
            diversity_scores = self._calculate_diversity_scores(chunk_vectors)
            
            # Calculate temporal relevance
            temporal_scores = self._calculate_temporal_relevance(chunks)
            
            # Combine scores with optimization goals
            combined_scores = self._combine_multi_modal_scores(
                similarities, diversity_scores, temporal_scores, request.optimization_goals
            )
            
            # Select optimal chunks
            chunk_scores = list(zip(chunks, combined_scores))
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Apply token budget constraint with optimization
            selected_chunks = self._optimize_token_allocation(
                chunk_scores, request.token_budget
            )
            
            return selected_chunks
            
        except Exception as e:
            logger.warning(f"Comprehensive analysis failed: {e}, falling back to balanced")
            return self._balanced_optimization_assembly(request, chunks)
    
    def _balanced_optimization_assembly(self, request: ContextOptimizationRequest,
                                      chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Balanced optimization strategy"""
        # Multi-factor scoring
        scored_chunks = []
        
        for chunk in chunks:
            score = 0.0
            
            # Relevance score (40% weight)
            relevance = chunk.get('relevance_score', 0.5)
            score += relevance * 0.4
            
            # Recency score (20% weight)
            recency = self._calculate_recency_score(chunk)
            score += recency * 0.2
            
            # Quality score (25% weight)
            quality = chunk.get('quality_score', 0.5)
            score += quality * 0.25
            
            # Usage popularity (15% weight)
            popularity = chunk.get('access_count', 0) / 100  # Normalize
            score += min(popularity, 1.0) * 0.15
            
            scored_chunks.append((chunk, score))
        
        # Sort by combined score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Select chunks within token budget
        selected_chunks = []
        token_count = 0
        
        for chunk, score in scored_chunks:
            chunk_tokens = chunk.get('token_count', 100)
            if token_count + chunk_tokens <= request.token_budget:
                chunk['combined_score'] = score
                selected_chunks.append(chunk)
                token_count += chunk_tokens
        
        return selected_chunks
    
    def _semantic_clustering_assembly(self, request: ContextOptimizationRequest,
                                    chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Semantic clustering-based assembly"""
        if not ML_AVAILABLE or len(chunks) < 5:
            return self._balanced_optimization_assembly(request, chunks)
        
        try:
            # Create text vectors
            chunk_texts = [chunk.get('content', '') for chunk in chunks]
            vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            vectors = vectorizer.fit_transform(chunk_texts)
            
            # Perform clustering
            clustering = DBSCAN(eps=0.5, min_samples=2, metric='cosine')
            cluster_labels = clustering.fit_predict(vectors.toarray())
            
            # Group chunks by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append((chunks[i], i))
            
            # Select diverse chunks from different clusters
            selected_chunks = []
            token_count = 0
            
            # Sort clusters by relevance
            cluster_relevance = {}
            for label, cluster_chunks in clusters.items():
                avg_relevance = statistics.mean(
                    chunk.get('relevance_score', 0.5) for chunk, _ in cluster_chunks
                )
                cluster_relevance[label] = avg_relevance
            
            sorted_clusters = sorted(cluster_relevance.items(), key=lambda x: x[1], reverse=True)
            
            # Select top chunks from each cluster
            for label, _ in sorted_clusters:
                cluster_chunks = clusters[label]
                cluster_chunks.sort(key=lambda x: x[0].get('relevance_score', 0), reverse=True)
                
                for chunk, _ in cluster_chunks[:2]:  # Top 2 from each cluster
                    chunk_tokens = chunk.get('token_count', 100)
                    if token_count + chunk_tokens <= request.token_budget:
                        selected_chunks.append(chunk)
                        token_count += chunk_tokens
                    
                    if len(selected_chunks) >= 15:  # Limit total chunks
                        break
                
                if len(selected_chunks) >= 15:
                    break
            
            return selected_chunks
            
        except Exception as e:
            logger.warning(f"Semantic clustering failed: {e}, falling back to balanced")
            return self._balanced_optimization_assembly(request, chunks)
    
    def _temporal_relevance_assembly(self, request: ContextOptimizationRequest,
                                   chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Temporal relevance-based assembly"""
        # Calculate temporal scores
        current_time = datetime.now()
        
        scored_chunks = []
        for chunk in chunks:
            # Base relevance
            base_score = chunk.get('relevance_score', 0.5)
            
            # Temporal boost
            created_time = chunk.get('created_at')
            updated_time = chunk.get('last_updated')
            
            temporal_score = 1.0
            if updated_time:
                days_since_update = (current_time - updated_time).days
                temporal_score = max(0.1, 1.0 - (days_since_update / 365))  # Decay over year
            
            # Access recency boost
            last_access = chunk.get('last_accessed')
            if last_access:
                hours_since_access = (current_time - last_access).total_seconds() / 3600
                access_boost = max(0.0, 1.0 - (hours_since_access / 168))  # Decay over week
                temporal_score += access_boost * 0.3
            
            combined_score = base_score * 0.7 + temporal_score * 0.3
            scored_chunks.append((chunk, combined_score))
        
        # Sort and select
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        selected_chunks = []
        token_count = 0
        
        for chunk, score in scored_chunks:
            chunk_tokens = chunk.get('token_count', 100)
            if token_count + chunk_tokens <= request.token_budget:
                chunk['temporal_score'] = score
                selected_chunks.append(chunk)
                token_count += chunk_tokens
        
        return selected_chunks
    
    def _calculate_diversity_scores(self, vectors) -> List[float]:
        """Calculate diversity scores for chunks"""
        if vectors.shape[0] < 2:
            return [1.0] * vectors.shape[0]
        
        similarities = cosine_similarity(vectors)
        diversity_scores = []
        
        for i in range(len(similarities)):
            # Average dissimilarity with other chunks
            dissimilarities = 1 - similarities[i]
            avg_dissimilarity = np.mean(dissimilarities)
            diversity_scores.append(avg_dissimilarity)
        
        return diversity_scores
    
    def _calculate_temporal_relevance(self, chunks: List[Dict[str, Any]]) -> List[float]:
        """Calculate temporal relevance scores"""
        current_time = datetime.now()
        scores = []
        
        for chunk in chunks:
            score = 0.5  # Default
            
            # Recency of creation
            created_at = chunk.get('created_at')
            if created_at:
                days_old = (current_time - created_at).days
                recency_score = max(0.0, 1.0 - (days_old / 365))
                score += recency_score * 0.3
            
            # Recency of updates
            updated_at = chunk.get('last_updated')
            if updated_at:
                days_since_update = (current_time - updated_at).days
                update_score = max(0.0, 1.0 - (days_since_update / 180))
                score += update_score * 0.3
            
            # Access frequency
            access_count = chunk.get('access_count', 0)
            if access_count > 0:
                frequency_score = min(1.0, access_count / 50)
                score += frequency_score * 0.2
            
            scores.append(min(1.0, score))
        
        return scores
    
    def _combine_multi_modal_scores(self, similarities: List[float], 
                                   diversity_scores: List[float],
                                   temporal_scores: List[float],
                                   goals: List[str]) -> List[float]:
        """Combine multiple scoring modalities based on optimization goals"""
        # Default weights
        weights = {
            'semantic': 0.5,
            'diversity': 0.2,
            'temporal': 0.3
        }
        
        # Adjust weights based on goals
        if 'accuracy' in goals:
            weights['semantic'] += 0.2
            weights['diversity'] -= 0.1
        
        if 'diversity' in goals:
            weights['diversity'] += 0.2
            weights['semantic'] -= 0.1
        
        if 'speed' in goals:
            weights['semantic'] += 0.1
            weights['diversity'] -= 0.05
            weights['temporal'] -= 0.05
        
        if 'recency' in goals or 'current' in goals:
            weights['temporal'] += 0.2
            weights['semantic'] -= 0.1
        
        # Combine scores
        combined_scores = []
        for i in range(len(similarities)):
            score = (similarities[i] * weights['semantic'] +
                    diversity_scores[i] * weights['diversity'] +
                    temporal_scores[i] * weights['temporal'])
            combined_scores.append(score)
        
        return combined_scores
    
    def _optimize_token_allocation(self, chunk_scores: List[Tuple], 
                                  token_budget: int) -> List[Dict[str, Any]]:
        """Optimize token allocation using dynamic programming approach"""
        # Sort by score density (score per token)
        scored_chunks = []
        for chunk, score in chunk_scores:
            tokens = chunk.get('token_count', 100)
            density = score / tokens if tokens > 0 else 0
            scored_chunks.append((chunk, score, tokens, density))
        
        # Sort by density
        scored_chunks.sort(key=lambda x: x[3], reverse=True)
        
        # Greedy selection with optimization
        selected = []
        remaining_budget = token_budget
        
        for chunk, score, tokens, density in scored_chunks:
            if tokens <= remaining_budget:
                selected.append(chunk)
                remaining_budget -= tokens
            elif remaining_budget > 50:  # Try to fill remaining space
                # Look for smaller chunks that fit
                continue
        
        return selected
    
    def _calculate_recency_score(self, chunk: Dict[str, Any]) -> float:
        """Calculate recency score for a chunk"""
        current_time = datetime.now()
        
        # Check last updated
        updated_at = chunk.get('last_updated')
        if updated_at:
            days_since_update = (current_time - updated_at).days
            return max(0.0, 1.0 - (days_since_update / 365))
        
        # Fallback to creation time
        created_at = chunk.get('created_at')
        if created_at:
            days_old = (current_time - created_at).days
            return max(0.0, 1.0 - (days_old / 730))  # 2 year decay
        
        return 0.5  # Default
    
    def _calculate_optimization_score(self, request: ContextOptimizationRequest,
                                    chunks: List[Dict[str, Any]]) -> float:
        """Calculate overall optimization score"""
        if not chunks:
            return 0.0
        
        # Base relevance score
        relevance_scores = [chunk.get('relevance_score', 0.5) for chunk in chunks]
        avg_relevance = statistics.mean(relevance_scores)
        
        # Token efficiency
        total_tokens = sum(chunk.get('token_count', 100) for chunk in chunks)
        token_efficiency = min(1.0, total_tokens / request.token_budget)
        
        # Diversity score
        diversity_score = self._calculate_selection_diversity(chunks)
        
        # Combine scores based on goals
        weights = [0.5, 0.3, 0.2]  # relevance, efficiency, diversity
        
        if 'accuracy' in request.optimization_goals:
            weights[0] += 0.2
        if 'speed' in request.optimization_goals:
            weights[1] += 0.1
        if 'diversity' in request.optimization_goals:
            weights[2] += 0.2
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        optimization_score = (
            avg_relevance * weights[0] +
            token_efficiency * weights[1] +
            diversity_score * weights[2]
        )
        
        return optimization_score
    
    def _calculate_selection_diversity(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate diversity of selected chunks"""
        if len(chunks) < 2:
            return 1.0
        
        if not ML_AVAILABLE:
            # Fallback: use topic diversity
            topics = [chunk.get('topic', 'general') for chunk in chunks]
            unique_topics = len(set(topics))
            return min(1.0, unique_topics / len(chunks))
        
        try:
            # Use TF-IDF similarity
            texts = [chunk.get('content', '') for chunk in chunks]
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            vectors = vectorizer.fit_transform(texts)
            
            similarities = cosine_similarity(vectors)
            avg_similarity = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            
            return 1.0 - avg_similarity  # Diversity is inverse of similarity
            
        except Exception:
            return 0.5  # Fallback

class AIContextOptimizer:
    """
    Main AI-powered context optimization engine
    """
    
    def __init__(self, db_manager, session_manager=None, ml_engine=None):
        self.db_manager = db_manager
        self.session_manager = session_manager
        self.ml_engine = ml_engine
        
        # Initialize AI components
        self.learning_engine = AdaptiveLearningEngine()
        self.context_assembler = IntelligentContextAssembler(self.learning_engine)
        
        # Optimization tracking
        self.optimization_history = deque(maxlen=10000)
        self.performance_metrics = defaultdict(list)
        
        # Background optimization
        self.is_running = False
        self.optimization_thread = None
        self.optimization_queue = asyncio.Queue(maxsize=1000)
        
        # Configuration
        self.default_token_budgets = {
            'haiku': 50000,
            'sonnet': 150000,
            'opus': 500000
        }
        
        logger.info("✅ AI Context Optimizer initialized")
    
    async def optimize_context(self, query: str, session_context: Dict[str, Any] = None,
                              model_type: str = 'sonnet', task_complexity: str = 'medium',
                              token_budget: Optional[int] = None,
                              optimization_goals: List[str] = None) -> Dict[str, Any]:
        """
        Main context optimization function
        """
        start_time = time.time()
        
        # Prepare optimization request
        request = ContextOptimizationRequest(
            request_id=str(uuid.uuid4()),
            query=query,
            session_context=session_context or {},
            model_type=model_type.lower(),
            task_complexity=task_complexity.lower(),
            token_budget=token_budget or self.default_token_budgets.get(model_type.lower(), 150000),
            optimization_goals=optimization_goals or ['accuracy', 'speed'],
            timestamp=datetime.now()
        )
        
        logger.info(f"🤖 Starting AI context optimization for: {query[:100]}...")
        
        try:
            # Get candidate chunks from database
            candidate_chunks = await self._get_candidate_chunks(request)
            
            if not candidate_chunks:
                return {
                    'success': False,
                    'error': 'No relevant chunks found',
                    'context': []
                }
            
            # Apply AI optimization
            optimized_context = self.context_assembler.assemble_context(request, candidate_chunks)
            
            # Track optimization
            self._track_optimization(request, optimized_context, time.time() - start_time)
            
            # Return optimized context
            return {
                'success': True,
                'context': optimized_context.chunks,
                'optimization_metadata': {
                    'context_id': optimized_context.context_id,
                    'optimization_score': optimized_context.optimization_score,
                    'strategy': optimized_context.optimization_strategy,
                    'confidence': optimized_context.confidence,
                    'token_usage': optimized_context.token_usage,
                    'token_budget': request.token_budget,
                    'optimization_time_ms': optimized_context.metadata['assembly_time_ms'],
                    'chunk_count': len(optimized_context.chunks)
                }
            }
            
        except Exception as e:
            logger.error(f"AI context optimization failed: {e}")
            return {
                'success': False,
                'error': f'Optimization failed: {str(e)}',
                'context': []
            }
    
    async def _get_candidate_chunks(self, request: ContextOptimizationRequest) -> List[Dict[str, Any]]:
        """Get candidate chunks for optimization"""
        try:
            # Use existing search functionality
            if hasattr(self.db_manager, 'search_chunks_dual_realm'):
                search_results = await self.db_manager.search_chunks_dual_realm(
                    query=request.query,
                    limit=50,  # Get more candidates for optimization
                    search_type='hybrid'
                )
                
                chunks = search_results.get('chunks', [])
                
                # Enhance chunks with additional metadata
                for chunk in chunks:
                    chunk['token_count'] = len(chunk.get('content', '').split()) * 1.3  # Rough estimate
                    chunk['quality_score'] = chunk.get('relevance_score', 0.5)  # Initial quality
                
                return chunks
            
            return []
            
        except Exception as e:
            logger.error(f"Failed to get candidate chunks: {e}")
            return []
    
    def _track_optimization(self, request: ContextOptimizationRequest, 
                          result: OptimizedContext, optimization_time: float):
        """Track optimization for performance analysis"""
        tracking_data = {
            'request_id': request.request_id,
            'query': request.query,
            'model_type': request.model_type,
            'task_complexity': request.task_complexity,
            'token_budget': request.token_budget,
            'optimization_goals': request.optimization_goals,
            'result_score': result.optimization_score,
            'result_confidence': result.confidence,
            'strategy_used': result.optimization_strategy,
            'chunks_selected': len(result.chunks),
            'tokens_used': result.token_usage,
            'optimization_time_ms': optimization_time * 1000,
            'timestamp': datetime.now()
        }
        
        self.optimization_history.append(tracking_data)
        
        # Update performance metrics
        model_key = f"{request.model_type}_{request.task_complexity}"
        self.performance_metrics[model_key].append({
            'score': result.optimization_score,
            'confidence': result.confidence,
            'efficiency': result.token_usage / request.token_budget,
            'timestamp': datetime.now()
        })
    
    async def provide_feedback(self, context_id: str, feedback: Dict[str, Any]) -> bool:
        """Provide feedback on optimization results"""
        try:
            # Find the optimization in history
            for record in reversed(list(self.optimization_history)):
                if record.get('context_id') == context_id:
                    # Update learning engine with feedback
                    # This would typically involve retraining or updating models
                    logger.info(f"📈 Received feedback for context {context_id}: {feedback}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")
            return False
    
    def get_optimization_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get optimization performance analytics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_optimizations = [
            opt for opt in self.optimization_history
            if opt['timestamp'] >= cutoff_time
        ]
        
        if not recent_optimizations:
            return {'message': 'No recent optimizations found'}
        
        # Calculate analytics
        analytics = {
            'total_optimizations': len(recent_optimizations),
            'avg_optimization_score': statistics.mean(opt['result_score'] for opt in recent_optimizations),
            'avg_confidence': statistics.mean(opt['result_confidence'] for opt in recent_optimizations),
            'avg_optimization_time_ms': statistics.mean(opt['optimization_time_ms'] for opt in recent_optimizations),
            'token_efficiency': statistics.mean(
                opt['tokens_used'] / opt['token_budget'] for opt in recent_optimizations
            ),
            'strategy_distribution': {},
            'model_performance': {}
        }
        
        # Strategy distribution
        strategy_counts = defaultdict(int)
        for opt in recent_optimizations:
            strategy_counts[opt['strategy_used']] += 1
        
        analytics['strategy_distribution'] = dict(strategy_counts)
        
        # Model performance breakdown
        model_performance = defaultdict(list)
        for opt in recent_optimizations:
            key = f"{opt['model_type']}_{opt['task_complexity']}"
            model_performance[key].append(opt['result_score'])
        
        for key, scores in model_performance.items():
            analytics['model_performance'][key] = {
                'avg_score': statistics.mean(scores),
                'optimization_count': len(scores)
            }
        
        return analytics
    
    def get_ai_status(self) -> Dict[str, Any]:
        """Get AI optimization system status"""
        return {
            'ai_optimizer_available': True,
            'learning_engine_active': True,
            'optimization_history_size': len(self.optimization_history),
            'ml_available': ML_AVAILABLE,
            'supported_strategies': list(self.context_assembler.assembly_strategies.keys()),
            'default_token_budgets': self.default_token_budgets,
            'performance_models_trained': len(self.learning_engine.performance_models),
            'optimization_patterns': len(self.learning_engine.optimization_patterns)
        }