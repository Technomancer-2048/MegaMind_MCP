#!/usr/bin/env python3
"""
Phase 6: ML-Enhanced MCP Server
Integrates real ML algorithms with Phase 5 advanced session functions
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import Phase 5 components
from phase5_enhanced_server import Phase5EnhancedMCPServer
from ml_semantic_engine import MLSemanticEngine

logger = logging.getLogger(__name__)

class Phase6MLEnhancedMCPServer(Phase5EnhancedMCPServer):
    """
    Phase 6 ML-Enhanced MCP Server
    Extends Phase 5 with real machine learning capabilities
    """
    
    def __init__(self, db_manager):
        super().__init__(db_manager)
        
        # Initialize ML Semantic Engine
        try:
            self.ml_engine = MLSemanticEngine(
                db_manager=self.db_manager,
                session_manager=self.session_manager
            )
            self.ml_engine_available = True
            logger.info("✅ ML Semantic Engine initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize ML Semantic Engine: {e}")
            self.ml_engine = None
            self.ml_engine_available = False
    
    def get_phase6_status(self) -> Dict[str, Any]:
        """Get Phase 6 ML enhancement status"""
        base_status = self.get_phase5_status()
        
        phase6_status = {
            **base_status,
            "phase6_ml_available": self.ml_engine_available,
            "ml_engine_status": self._get_ml_engine_status(),
            "enhanced_functions": self._get_enhanced_functions_list(),
            "ml_capabilities": self._get_ml_capabilities()
        }
        
        return phase6_status
    
    def _get_ml_engine_status(self) -> Dict[str, Any]:
        """Get detailed ML engine status"""
        if not self.ml_engine_available:
            return {
                "available": False,
                "error": "ML engine not initialized"
            }
        
        return {
            "available": True,
            "sklearn_available": hasattr(self.ml_engine, 'SKLEARN_AVAILABLE') and self.ml_engine.SKLEARN_AVAILABLE,
            "vectorizer_initialized": self.ml_engine.vectorizer is not None,
            "models_loaded": len(self.ml_engine.models),
            "feature_cache_size": len(self.ml_engine.feature_cache)
        }
    
    def _get_enhanced_functions_list(self) -> List[str]:
        """List functions enhanced with ML capabilities"""
        return [
            "mcp__megamind__session_semantic_similarity",
            "mcp__megamind__session_semantic_clustering", 
            "mcp__megamind__session_semantic_insights",
            "mcp__megamind__session_semantic_recommendations",
            "mcp__megamind__session_analytics_dashboard"
        ]
    
    def _get_ml_capabilities(self) -> Dict[str, bool]:
        """Get available ML capabilities"""
        if not self.ml_engine_available:
            return {}
        
        try:
            # Check for scikit-learn availability
            from sklearn.feature_extraction.text import TfidfVectorizer
            sklearn_available = True
        except ImportError:
            sklearn_available = False
        
        return {
            "real_semantic_similarity": sklearn_available,
            "ml_clustering": sklearn_available,
            "topic_modeling": sklearn_available,
            "anomaly_detection": sklearn_available,
            "predictive_analytics": sklearn_available
        }
    
    # ================================================================
    # ENHANCED MCP FUNCTION HANDLERS WITH REAL ML
    # ================================================================
    
    def handle_session_semantic_similarity_ml(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced semantic similarity with real ML algorithms
        Replaces Phase 5 placeholder with actual ML implementation
        """
        try:
            reference_session_id = args.get('reference_session_id')
            user_id = args.get('user_id')
            similarity_threshold = args.get('similarity_threshold', 0.6)
            max_results = args.get('max_results', 10)
            include_archived = args.get('include_archived', False)
            analysis_depth = args.get('analysis_depth', 'content')
            
            # Get reference session
            reference_session = self.session_manager.get_session(reference_session_id)
            if not reference_session:
                return {"success": False, "error": f"Reference session {reference_session_id} not found"}
            
            # Get sessions to compare
            if user_id:
                comparison_sessions = self.session_manager.list_user_sessions(user_id, reference_session.realm_id, limit=1000)
            else:
                comparison_sessions = self.session_manager.list_user_sessions(
                    reference_session.user_id, reference_session.realm_id, limit=1000
                )
            
            # Filter sessions
            if not include_archived:
                comparison_sessions = [s for s in comparison_sessions if s.session_state.value != 'archived']
            
            # Remove reference session
            comparison_sessions = [s for s in comparison_sessions if s.session_id != reference_session_id]
            
            # Use ML engine for real similarity calculation
            if self.ml_engine_available:
                similarities = self.ml_engine.calculate_session_similarities_real(
                    reference_session, comparison_sessions, analysis_depth
                )
            else:
                # Fallback to Phase 5 implementation
                similarities = self.advanced_functions_part2._calculate_session_similarities(
                    reference_session, comparison_sessions, analysis_depth
                )
            
            # Filter by threshold and limit results
            filtered_similarities = [
                sim for sim in similarities if sim['similarity_score'] >= similarity_threshold
            ]
            filtered_similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            limited_similarities = filtered_similarities[:max_results]
            
            return {
                "success": True,
                "ml_enhanced": self.ml_engine_available,
                "reference_session_id": reference_session_id,
                "analysis_depth": analysis_depth,
                "similarity_threshold": similarity_threshold,
                "total_sessions_compared": len(comparison_sessions),
                "similar_sessions_found": len(limited_similarities),
                "similar_sessions": limited_similarities,
                "analysis_metadata": {
                    "analyzed_at": datetime.now().isoformat(),
                    "ml_method": "tfidf_cosine" if self.ml_engine_available else "placeholder",
                    "include_archived": include_archived,
                    "max_results": max_results
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate ML-enhanced semantic similarity: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_semantic_clustering_ml(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced semantic clustering with real ML algorithms
        Replaces Phase 5 placeholder with actual ML implementation
        """
        try:
            user_id = args.get('user_id')
            realm_id = args.get('realm_id', 'PROJECT')
            clustering_method = args.get('clustering_method', 'kmeans')
            num_clusters = args.get('num_clusters', 5)
            feature_type = args.get('feature_type', 'tfidf')
            min_sessions = args.get('min_sessions', 3)
            time_period = args.get('time_period')
            
            # Get sessions for clustering
            sessions = self.session_manager.list_user_sessions(user_id, realm_id, limit=1000)
            
            # Apply time period filter if specified
            if time_period:
                period_days = self.advanced_functions_part2._parse_time_period(time_period)
                cutoff_date = datetime.now() - timedelta(days=period_days)
                sessions = [s for s in sessions if s.created_at and s.created_at >= cutoff_date]
            
            # Filter sessions with minimum content
            sessions = [s for s in sessions if s.total_entries >= min_sessions]
            
            if len(sessions) < num_clusters:
                return {
                    "success": False,
                    "error": f"Not enough sessions ({len(sessions)}) for {num_clusters} clusters"
                }
            
            # Use ML engine for real clustering
            if self.ml_engine_available:
                clusters = self.ml_engine.perform_ml_clustering(
                    sessions, clustering_method, num_clusters, feature_type
                )
                ml_enhanced = True
            else:
                # Fallback to Phase 5 implementation
                feature_vectors = self.advanced_functions_part2._generate_clustering_features(sessions, feature_type)
                clusters = self.advanced_functions_part2._perform_clustering(
                    feature_vectors, sessions, clustering_method, num_clusters
                )
                ml_enhanced = False
            
            # Enhanced cluster analysis
            if self.ml_engine_available:
                for cluster in clusters:
                    cluster['enhanced_analysis'] = self._enhance_cluster_analysis(cluster, sessions)
            
            return {
                "success": True,
                "ml_enhanced": ml_enhanced,
                "user_id": user_id,
                "realm_id": realm_id,
                "clustering_method": clustering_method,
                "feature_type": feature_type,
                "total_sessions": len(sessions),
                "num_clusters": len(clusters),
                "min_sessions_per_cluster": min_sessions,
                "clusters": clusters,
                "clustering_metadata": {
                    "clustered_at": datetime.now().isoformat(),
                    "ml_method": clustering_method if ml_enhanced else "simple_grouping",
                    "time_period": time_period,
                    "feature_dimensions": getattr(clusters[0], 'feature_metadata', {}).get('feature_dimensions', 0) if clusters else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to perform ML-enhanced clustering: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_semantic_insights_ml(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced semantic insights with real ML topic modeling
        Replaces Phase 5 placeholder with actual ML implementation
        """
        try:
            session_ids = args.get('session_ids', [])
            user_id = args.get('user_id')
            insight_types = args.get('insight_types', ['topics', 'trends', 'anomalies'])
            topic_modeling = args.get('topic_modeling', True)
            trend_analysis = args.get('trend_analysis', True)
            anomaly_detection = args.get('anomaly_detection', True)
            time_granularity = args.get('time_granularity', 'daily')
            num_topics = args.get('num_topics', 5)
            topic_method = args.get('topic_method', 'lda')
            
            # Collect sessions for analysis
            analysis_sessions = []
            if session_ids:
                for session_id in session_ids:
                    session = self.session_manager.get_session(session_id)
                    if session:
                        analysis_sessions.append(session)
            elif user_id:
                analysis_sessions = self.session_manager.list_user_sessions(user_id, 'PROJECT', limit=1000)
            
            insights = {
                "analysis_scope": {
                    "session_ids": session_ids if session_ids else "all_user_sessions",
                    "user_id": user_id,
                    "total_sessions": len(analysis_sessions),
                    "insight_types": insight_types,
                    "time_granularity": time_granularity,
                    "ml_enhanced": self.ml_engine_available
                }
            }
            
            # Real topic modeling insights
            if topic_modeling and 'topics' in insight_types and self.ml_engine_available:
                insights["topic_insights"] = self.ml_engine.generate_topic_insights_real(
                    analysis_sessions, num_topics, topic_method
                )
            elif topic_modeling and 'topics' in insight_types:
                insights["topic_insights"] = self.advanced_functions_part2._generate_topic_insights(analysis_sessions)
            
            # Enhanced trend analysis (using existing implementation but with ML features)
            if trend_analysis and 'trends' in insight_types:
                insights["trend_insights"] = self._generate_enhanced_trend_insights(
                    analysis_sessions, time_granularity
                )
            
            # Real anomaly detection
            if anomaly_detection and 'anomalies' in insight_types and self.ml_engine_available:
                insights["anomaly_insights"] = self.ml_engine.detect_session_anomalies(analysis_sessions)
            elif anomaly_detection and 'anomalies' in insight_types:
                insights["anomaly_insights"] = self.advanced_functions_part2._generate_anomaly_insights(analysis_sessions)
            
            # Enhanced semantic patterns
            insights["semantic_patterns"] = self._generate_enhanced_semantic_patterns(analysis_sessions)
            insights["content_evolution"] = self._analyze_enhanced_content_evolution(analysis_sessions)
            
            return {
                "success": True,
                "ml_enhanced": self.ml_engine_available,
                "insights": insights,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate ML-enhanced semantic insights: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_session_analytics_dashboard_ml(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced analytics dashboard with ML-powered insights
        """
        try:
            # Get base analytics from Phase 5
            base_analytics = self.advanced_functions_part2.handle_session_analytics_dashboard(args)
            
            if not base_analytics.get('success'):
                return base_analytics
            
            # Add ML-enhanced analytics if available
            if self.ml_engine_available:
                user_id = args.get('user_id')
                realm_id = args.get('realm_id', 'PROJECT')
                time_period = args.get('time_period', '30d')
                
                # Get sessions for ML analysis
                period_days = self.advanced_functions_part2._parse_time_period(time_period)
                cutoff_date = datetime.now() - timedelta(days=period_days)
                sessions = self.session_manager.list_user_sessions(user_id, realm_id, limit=1000)
                period_sessions = [s for s in sessions if s.created_at and s.created_at >= cutoff_date]
                
                # Add ML-powered insights
                ml_insights = {
                    "session_similarity_network": self._generate_similarity_network(period_sessions),
                    "automated_clustering": self._generate_automated_clustering_insights(period_sessions),
                    "anomaly_detection": self.ml_engine.detect_session_anomalies(period_sessions),
                    "predictive_insights": self._generate_predictive_insights(period_sessions),
                    "ml_recommendations": self._generate_ml_recommendations(period_sessions)
                }
                
                base_analytics["analytics"]["ml_enhanced_insights"] = ml_insights
                base_analytics["ml_enhanced"] = True
            
            return base_analytics
            
        except Exception as e:
            logger.error(f"Failed to generate ML-enhanced analytics dashboard: {e}")
            return {"success": False, "error": str(e)}
    
    # ================================================================
    # ML ENHANCEMENT HELPER METHODS
    # ================================================================
    
    def _enhance_cluster_analysis(self, cluster: Dict[str, Any], all_sessions: List) -> Dict[str, Any]:
        """Enhance cluster analysis with ML insights"""
        if not self.ml_engine_available:
            return {"enhancement": "ML engine not available"}
        
        cluster_sessions = cluster.get('sessions', [])
        if not cluster_sessions:
            return {"enhancement": "No sessions in cluster"}
        
        # Generate topic model for cluster
        try:
            topic_insights = self.ml_engine.generate_topic_insights_real(cluster_sessions, num_topics=3)
            
            # Calculate cluster coherence
            coherence_score = self._calculate_cluster_coherence(cluster_sessions)
            
            return {
                "dominant_topics": topic_insights.get('topics_discovered', []),
                "cluster_coherence": coherence_score,
                "cluster_diversity": self._calculate_cluster_diversity(cluster_sessions),
                "temporal_patterns": self._analyze_cluster_temporal_patterns(cluster_sessions)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to enhance cluster analysis: {e}")
            return {"enhancement_error": str(e)}
    
    def _calculate_cluster_coherence(self, sessions: List) -> float:
        """Calculate semantic coherence of a cluster"""
        if len(sessions) < 2:
            return 1.0
        
        try:
            # Calculate pairwise similarities within cluster
            similarities = []
            for i in range(len(sessions)):
                for j in range(i + 1, len(sessions)):
                    sim_result = self.ml_engine.calculate_session_similarities_real(
                        sessions[i], [sessions[j]], 'content'
                    )
                    if sim_result:
                        similarities.append(sim_result[0]['similarity_score'])
            
            return float(np.mean(similarities)) if similarities else 0.0
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate cluster coherence: {e}")
            return 0.0
    
    def _calculate_cluster_diversity(self, sessions: List) -> float:
        """Calculate diversity within a cluster"""
        if len(sessions) < 2:
            return 0.0
        
        # Simple diversity based on session characteristics
        priorities = [session.priority.value for session in sessions]
        states = [session.session_state.value for session in sessions]
        
        priority_diversity = len(set(priorities)) / len(priorities)
        state_diversity = len(set(states)) / len(states)
        
        return (priority_diversity + state_diversity) / 2
    
    def _analyze_cluster_temporal_patterns(self, sessions: List) -> Dict[str, Any]:
        """Analyze temporal patterns within a cluster"""
        if not sessions:
            return {}
        
        creation_times = [s.created_at for s in sessions if s.created_at]
        if not creation_times:
            return {}
        
        # Calculate temporal statistics
        time_span = (max(creation_times) - min(creation_times)).total_seconds() / 3600  # hours
        avg_interval = time_span / len(creation_times) if len(creation_times) > 1 else 0
        
        return {
            "temporal_span_hours": time_span,
            "average_creation_interval_hours": avg_interval,
            "creation_frequency": len(creation_times) / max(time_span / 24, 1)  # sessions per day
        }
    
    def _generate_enhanced_trend_insights(self, sessions: List, granularity: str) -> Dict[str, Any]:
        """Generate enhanced trend insights with ML"""
        base_trends = self.advanced_functions_part2._generate_trend_insights(sessions, granularity)
        
        if not self.ml_engine_available:
            return base_trends
        
        # Add ML-based trend analysis
        try:
            # Analyze session activity over time
            session_timeline = []
            for session in sessions:
                if session.created_at:
                    session_timeline.append({
                        'date': session.created_at,
                        'entries': session.total_entries or 0,
                        'operations': session.total_operations or 0,
                        'performance': session.performance_score or 0.5
                    })
            
            # Sort by date
            session_timeline.sort(key=lambda x: x['date'])
            
            if len(session_timeline) > 3:
                # Calculate trends
                entry_trend = self._calculate_trend([s['entries'] for s in session_timeline])
                performance_trend = self._calculate_trend([s['performance'] for s in session_timeline])
                
                base_trends.update({
                    "ml_enhanced": True,
                    "activity_trend": entry_trend,
                    "performance_trend": performance_trend,
                    "trend_confidence": self._calculate_trend_confidence(session_timeline)
                })
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate enhanced trends: {e}")
        
        return base_trends
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend direction and strength"""
        if len(values) < 3:
            return {"direction": "insufficient_data", "strength": 0.0}
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        try:
            correlation = np.corrcoef(x, values)[0, 1] if len(values) > 1 else 0
            
            if correlation > 0.3:
                direction = "increasing"
            elif correlation < -0.3:
                direction = "decreasing"
            else:
                direction = "stable"
            
            return {
                "direction": direction,
                "strength": abs(correlation),
                "correlation": correlation
            }
            
        except Exception:
            return {"direction": "unknown", "strength": 0.0}
    
    def _calculate_trend_confidence(self, timeline: List[Dict]) -> float:
        """Calculate confidence in trend analysis"""
        if len(timeline) < 5:
            return 0.3  # Low confidence with limited data
        elif len(timeline) < 10:
            return 0.6  # Medium confidence
        else:
            return 0.8  # High confidence with sufficient data
    
    def _generate_enhanced_semantic_patterns(self, sessions: List) -> Dict[str, Any]:
        """Generate enhanced semantic patterns using ML"""
        base_patterns = self.advanced_functions_part2._generate_semantic_patterns(sessions)
        
        if not self.ml_engine_available or len(sessions) < 3:
            return base_patterns
        
        try:
            # Perform clustering to find semantic patterns
            clusters = self.ml_engine.perform_ml_clustering(sessions, 'kmeans', min(5, len(sessions)//2), 'tfidf')
            
            # Analyze patterns across clusters
            pattern_analysis = {
                "semantic_clusters": len(clusters),
                "cluster_themes": [],
                "cross_cluster_patterns": self._analyze_cross_cluster_patterns(clusters)
            }
            
            # Extract themes from each cluster
            for cluster in clusters:
                cluster_sessions = cluster.get('sessions', [])
                if cluster_sessions:
                    theme_analysis = self._extract_cluster_theme(cluster_sessions)
                    pattern_analysis["cluster_themes"].append({
                        "cluster_id": cluster.get('cluster_id'),
                        "session_count": len(cluster_sessions),
                        "dominant_theme": theme_analysis
                    })
            
            base_patterns.update({
                "ml_enhanced": True,
                "pattern_analysis": pattern_analysis
            })
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate enhanced semantic patterns: {e}")
        
        return base_patterns
    
    def _analyze_cross_cluster_patterns(self, clusters: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns across different clusters"""
        if len(clusters) < 2:
            return {"analysis": "Insufficient clusters for cross-analysis"}
        
        # Analyze cluster relationships
        cluster_sizes = [cluster.get('session_count', 0) for cluster in clusters]
        cluster_methods = [cluster.get('cluster_method', 'unknown') for cluster in clusters]
        
        return {
            "size_distribution": {
                "largest_cluster": max(cluster_sizes),
                "smallest_cluster": min(cluster_sizes),
                "average_size": sum(cluster_sizes) / len(cluster_sizes)
            },
            "clustering_quality": self._assess_clustering_quality(clusters),
            "cluster_separation": "good" if len(set(cluster_methods)) == 1 else "mixed"
        }
    
    def _extract_cluster_theme(self, sessions: List) -> Dict[str, Any]:
        """Extract dominant theme from a cluster of sessions"""
        if not sessions:
            return {"theme": "empty_cluster"}
        
        # Analyze session names and content for themes
        session_names = [session.session_name or "" for session in sessions]
        common_words = []
        
        for name in session_names:
            words = name.lower().split()
            common_words.extend([word for word in words if len(word) > 3])
        
        if common_words:
            word_counts = Counter(common_words)
            top_words = word_counts.most_common(3)
            theme = "_".join([word for word, _ in top_words])
        else:
            theme = "unthemed"
        
        return {
            "theme": theme,
            "confidence": min(1.0, len(common_words) / 10),  # Confidence based on word frequency
            "characteristic_words": [word for word, _ in word_counts.most_common(5)]
        }
    
    def _assess_clustering_quality(self, clusters: List[Dict]) -> str:
        """Assess the quality of clustering results"""
        if not clusters:
            return "no_clusters"
        
        cluster_sizes = [cluster.get('session_count', 0) for cluster in clusters]
        
        # Check for balanced clusters
        size_variance = np.var(cluster_sizes) if cluster_sizes else 0
        avg_size = np.mean(cluster_sizes) if cluster_sizes else 0
        
        if size_variance / (avg_size + 1) < 0.5:
            return "well_balanced"
        elif size_variance / (avg_size + 1) < 1.0:
            return "moderately_balanced"
        else:
            return "imbalanced"
    
    def _analyze_enhanced_content_evolution(self, sessions: List) -> Dict[str, Any]:
        """Analyze content evolution using ML techniques"""
        base_evolution = self.advanced_functions_part2._analyze_content_evolution(sessions)
        
        if not self.ml_engine_available or len(sessions) < 5:
            return base_evolution
        
        try:
            # Sort sessions by creation time
            sorted_sessions = sorted([s for s in sessions if s.created_at], key=lambda x: x.created_at)
            
            if len(sorted_sessions) < 5:
                return base_evolution
            
            # Analyze evolution patterns
            evolution_metrics = []
            window_size = min(5, len(sorted_sessions) // 3)
            
            for i in range(0, len(sorted_sessions) - window_size + 1, window_size):
                window_sessions = sorted_sessions[i:i + window_size]
                
                # Calculate window characteristics
                avg_complexity = np.mean([s.total_operations or 0 for s in window_sessions])
                avg_performance = np.mean([s.performance_score or 0.5 for s in window_sessions])
                
                evolution_metrics.append({
                    "window_start": i,
                    "complexity": avg_complexity,
                    "performance": avg_performance,
                    "session_count": len(window_sessions)
                })
            
            # Calculate evolution trends
            complexity_trend = self._calculate_trend([m['complexity'] for m in evolution_metrics])
            performance_trend = self._calculate_trend([m['performance'] for m in evolution_metrics])
            
            base_evolution.update({
                "ml_enhanced": True,
                "evolution_windows": evolution_metrics,
                "complexity_evolution": complexity_trend,
                "performance_evolution": performance_trend,
                "evolution_quality": self._assess_evolution_quality(evolution_metrics)
            })
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to analyze enhanced content evolution: {e}")
        
        return base_evolution
    
    def _assess_evolution_quality(self, evolution_metrics: List[Dict]) -> str:
        """Assess the quality of content evolution"""
        if len(evolution_metrics) < 2:
            return "insufficient_data"
        
        # Check for improvement trends
        complexity_values = [m['complexity'] for m in evolution_metrics]
        performance_values = [m['performance'] for m in evolution_metrics]
        
        complexity_improving = complexity_values[-1] > complexity_values[0]
        performance_improving = performance_values[-1] > performance_values[0]
        
        if complexity_improving and performance_improving:
            return "positive_evolution"
        elif performance_improving:
            return "performance_improving"
        elif complexity_improving:
            return "complexity_increasing"
        else:
            return "declining_trends"
    
    # ================================================================
    # NEW ML-POWERED HELPER METHODS
    # ================================================================
    
    def _generate_similarity_network(self, sessions: List) -> Dict[str, Any]:
        """Generate a similarity network of sessions"""
        if not self.ml_engine_available or len(sessions) < 3:
            return {"network": "insufficient_data"}
        
        try:
            # Calculate pairwise similarities
            similarity_matrix = []
            session_pairs = []
            
            for i in range(len(sessions)):
                for j in range(i + 1, len(sessions)):
                    sim_result = self.ml_engine.calculate_session_similarities_real(
                        sessions[i], [sessions[j]], 'content'
                    )
                    if sim_result:
                        similarity_score = sim_result[0]['similarity_score']
                        if similarity_score > 0.3:  # Only include meaningful similarities
                            session_pairs.append({
                                'session1': sessions[i].session_id,
                                'session2': sessions[j].session_id,
                                'similarity': similarity_score
                            })
            
            return {
                "network_nodes": len(sessions),
                "network_edges": len(session_pairs),
                "high_similarity_pairs": [p for p in session_pairs if p['similarity'] > 0.7],
                "network_density": len(session_pairs) / (len(sessions) * (len(sessions) - 1) / 2) if len(sessions) > 1 else 0
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate similarity network: {e}")
            return {"network_error": str(e)}
    
    def _generate_automated_clustering_insights(self, sessions: List) -> Dict[str, Any]:
        """Generate automated clustering insights"""
        if not self.ml_engine_available or len(sessions) < 5:
            return {"clustering": "insufficient_data"}
        
        try:
            # Try different clustering methods
            clustering_results = {}
            
            methods = ['kmeans', 'dbscan', 'hierarchical']
            for method in methods:
                try:
                    clusters = self.ml_engine.perform_ml_clustering(
                        sessions, method, min(5, len(sessions)//2), 'tfidf'
                    )
                    clustering_results[method] = {
                        "num_clusters": len(clusters),
                        "quality_score": self._evaluate_clustering_quality(clusters, sessions)
                    }
                except Exception as e:
                    clustering_results[method] = {"error": str(e)}
            
            # Find best clustering method
            best_method = max(
                clustering_results.keys(),
                key=lambda k: clustering_results[k].get('quality_score', 0)
            )
            
            return {
                "clustering_comparison": clustering_results,
                "recommended_method": best_method,
                "optimal_clusters": clustering_results[best_method].get('num_clusters', 0)
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate automated clustering insights: {e}")
            return {"clustering_error": str(e)}
    
    def _evaluate_clustering_quality(self, clusters: List[Dict], sessions: List) -> float:
        """Evaluate the quality of a clustering result"""
        if not clusters or len(clusters) < 2:
            return 0.0
        
        try:
            # Calculate quality metrics
            cluster_sizes = [cluster.get('session_count', 0) for cluster in clusters]
            
            # Balance score (prefer balanced clusters)
            size_variance = np.var(cluster_sizes)
            avg_size = np.mean(cluster_sizes)
            balance_score = 1.0 / (1.0 + size_variance / (avg_size + 1))
            
            # Coverage score (prefer high coverage)
            total_clustered = sum(cluster_sizes)
            coverage_score = total_clustered / len(sessions)
            
            # Combined quality score
            quality_score = (balance_score * 0.6 + coverage_score * 0.4)
            
            return quality_score
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to evaluate clustering quality: {e}")
            return 0.0
    
    def _generate_predictive_insights(self, sessions: List) -> Dict[str, Any]:
        """Generate predictive insights about future session patterns"""
        if len(sessions) < 10:
            return {"prediction": "insufficient_historical_data"}
        
        try:
            # Analyze recent trends
            recent_sessions = sorted(
                [s for s in sessions if s.created_at],
                key=lambda x: x.created_at
            )[-10:]  # Last 10 sessions
            
            # Calculate trend indicators
            activity_trend = [s.total_operations or 0 for s in recent_sessions]
            performance_trend = [s.performance_score or 0.5 for s in recent_sessions]
            
            # Simple predictive analysis
            avg_activity_change = (activity_trend[-1] - activity_trend[0]) / len(activity_trend)
            avg_performance_change = (performance_trend[-1] - performance_trend[0]) / len(performance_trend)
            
            # Generate predictions
            predictions = {
                "activity_forecast": {
                    "trend": "increasing" if avg_activity_change > 0 else "decreasing",
                    "confidence": min(0.8, abs(avg_activity_change) / 10),
                    "next_session_activity": max(0, activity_trend[-1] + avg_activity_change)
                },
                "performance_forecast": {
                    "trend": "improving" if avg_performance_change > 0 else "declining",
                    "confidence": min(0.8, abs(avg_performance_change)),
                    "next_session_performance": max(0, min(1, performance_trend[-1] + avg_performance_change))
                },
                "recommendations": self._generate_predictive_recommendations(
                    avg_activity_change, avg_performance_change
                )
            }
            
            return predictions
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate predictive insights: {e}")
            return {"prediction_error": str(e)}
    
    def _generate_predictive_recommendations(self, activity_change: float, performance_change: float) -> List[str]:
        """Generate recommendations based on predictive trends"""
        recommendations = []
        
        if activity_change > 0 and performance_change < 0:
            recommendations.append("Consider session optimization - activity increasing but performance declining")
        elif activity_change < 0:
            recommendations.append("Monitor user engagement - activity trending downward")
        elif performance_change > 0:
            recommendations.append("Current workflow is effective - performance improving")
        
        if abs(activity_change) > 5:
            recommendations.append("Significant activity changes detected - review session patterns")
        
        return recommendations if recommendations else ["No specific recommendations based on current trends"]
    
    def _generate_ml_recommendations(self, sessions: List) -> List[Dict[str, Any]]:
        """Generate ML-powered recommendations"""
        if not self.ml_engine_available or len(sessions) < 5:
            return [{"recommendation": "insufficient_data_for_ml_recommendations"}]
        
        try:
            recommendations = []
            
            # Clustering-based recommendations
            clusters = self.ml_engine.perform_ml_clustering(sessions, 'kmeans', min(3, len(sessions)//3), 'tfidf')
            if len(clusters) > 1:
                largest_cluster = max(clusters, key=lambda c: c.get('session_count', 0))
                recommendations.append({
                    "type": "clustering",
                    "recommendation": f"Focus on {largest_cluster['cluster_method']} patterns",
                    "confidence": 0.7,
                    "details": f"Largest cluster contains {largest_cluster.get('session_count', 0)} sessions"
                })
            
            # Anomaly-based recommendations
            anomalies = self.ml_engine.detect_session_anomalies(sessions)
            if anomalies.get('anomalies_detected', 0) > 0:
                recommendations.append({
                    "type": "anomaly_detection",
                    "recommendation": f"Review {anomalies['anomalies_detected']} anomalous sessions",
                    "confidence": 0.8,
                    "details": "Outlier sessions may indicate issues or opportunities"
                })
            
            # Topic-based recommendations
            if len(sessions) >= 5:
                topics = self.ml_engine.generate_topic_insights_real(sessions, 3)
                if topics.get('topics_discovered'):
                    recommendations.append({
                        "type": "topic_modeling",
                        "recommendation": "Consider organizing sessions by dominant topics",
                        "confidence": 0.6,
                        "details": f"Identified {len(topics['topics_discovered'])} main topics"
                    })
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate ML recommendations: {e}")
            return [{"recommendation_error": str(e)}]
    
    # ================================================================
    # OVERRIDE PHASE 5 FUNCTIONS WITH ML ENHANCEMENTS
    # ================================================================
    
    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced tool call handler with ML capabilities
        Routes to ML-enhanced versions when available
        """
        # ML-enhanced function routing
        ml_enhanced_functions = {
            'mcp__megamind__session_semantic_similarity': self.handle_session_semantic_similarity_ml,
            'mcp__megamind__session_semantic_clustering': self.handle_session_semantic_clustering_ml,
            'mcp__megamind__session_semantic_insights': self.handle_session_semantic_insights_ml,
            'mcp__megamind__session_analytics_dashboard': self.handle_session_analytics_dashboard_ml
        }
        
        if tool_name in ml_enhanced_functions and self.ml_engine_available:
            logger.info(f"🤖 Using ML-enhanced version of {tool_name}")
            return ml_enhanced_functions[tool_name](args)
        else:
            # Fall back to Phase 5 implementation
            return super().handle_tool_call(tool_name, args)