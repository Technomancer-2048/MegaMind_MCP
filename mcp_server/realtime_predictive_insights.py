#!/usr/bin/env python3
"""
Phase 7: Real-time Predictive Insights Engine
Advanced predictive capabilities with real-time recommendations and forecasting
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import statistics
import threading
from dataclasses import dataclass, asdict
import uuid

# ML imports for real-time predictions
try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available - using fallback predictive insights")

logger = logging.getLogger(__name__)

@dataclass
class PredictiveInsight:
    """Real-time predictive insight data structure"""
    insight_id: str
    insight_type: str  # 'success_prediction', 'performance_forecast', 'risk_assessment', 'recommendation'
    target: str  # session_id, user_id, or 'system'
    prediction: Any
    confidence: float
    reasoning: List[str]
    recommended_actions: List[str]
    timestamp: datetime
    expires_at: Optional[datetime] = None
    ml_features: Optional[Dict[str, float]] = None
    impact_score: Optional[float] = None

@dataclass
class RealtimeRecommendation:
    """Real-time recommendation data structure"""
    recommendation_id: str
    category: str  # 'performance', 'workflow', 'resource', 'quality'
    priority: str  # 'low', 'medium', 'high', 'urgent'
    title: str
    description: str
    context: Dict[str, Any]
    estimated_impact: str  # 'positive', 'neutral', 'negative'
    implementation_effort: str  # 'low', 'medium', 'high'
    timestamp: datetime
    valid_until: Optional[datetime] = None

class RealTimePredictiveEngine:
    """
    Real-time predictive insights engine providing live recommendations and forecasts
    """
    
    def __init__(self, session_manager, ml_engine=None, analytics_engine=None):
        self.session_manager = session_manager
        self.ml_engine = ml_engine
        self.analytics_engine = analytics_engine
        
        # Insight tracking
        self.active_insights = {}  # insight_id -> PredictiveInsight
        self.insight_history = deque(maxlen=10000)
        self.recommendations = {}  # recommendation_id -> RealtimeRecommendation
        
        # Prediction models
        self.prediction_models = {}
        self.feature_scalers = {}
        self.model_last_trained = {}
        
        # Real-time data streams
        self.session_performance_stream = defaultdict(lambda: deque(maxlen=100))
        self.user_behavior_stream = defaultdict(lambda: deque(maxlen=500))
        self.system_metrics_stream = deque(maxlen=1000)
        
        # Insight generation settings
        self.insight_refresh_interval = 30  # seconds
        self.recommendation_ttl_hours = 24
        self.confidence_threshold = 0.6
        
        # Background processing
        self.is_running = False
        self.processing_thread = None
        self.lock = threading.RLock()
        
        # Initialize prediction models
        self._initialize_prediction_models()
        
        logger.info("✅ Real-time Predictive Insights Engine initialized")
    
    def _initialize_prediction_models(self):
        """Initialize real-time prediction models"""
        if not ML_AVAILABLE:
            logger.warning("ML not available - using fallback prediction models")
            return
        
        try:
            self.prediction_models = {
                'session_success_predictor': RandomForestClassifier(n_estimators=50, random_state=42),
                'performance_forecaster': RandomForestRegressor(n_estimators=50, random_state=42),
                'completion_time_predictor': LinearRegression(),
                'quality_predictor': RandomForestRegressor(n_estimators=30, random_state=42),
                'resource_usage_predictor': LinearRegression()
            }
            
            self.feature_scalers = {
                model_name: StandardScaler() 
                for model_name in self.prediction_models.keys()
            }
            
            logger.info("✅ Real-time prediction models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize prediction models: {e}")
    
    def start_predictions(self):
        """Start real-time prediction generation"""
        if self.is_running:
            logger.warning("⚠️ Predictive insights already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._prediction_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("🚀 Real-time Predictive Insights started")
    
    def stop_predictions(self):
        """Stop real-time prediction generation"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("⏹️ Real-time Predictive Insights stopped")
    
    def add_session_data(self, session_id: str, performance_data: Dict[str, Any]):
        """Add real-time session performance data"""
        with self.lock:
            self.session_performance_stream[session_id].append({
                'timestamp': datetime.now(),
                'data': performance_data
            })
            
            # Trigger immediate predictions for this session if significant change
            if self._is_significant_change(session_id, performance_data):
                self._generate_session_insights(session_id)
    
    def add_user_behavior(self, user_id: str, behavior_data: Dict[str, Any]):
        """Add real-time user behavior data"""
        with self.lock:
            self.user_behavior_stream[user_id].append({
                'timestamp': datetime.now(),
                'data': behavior_data
            })
    
    def add_system_metrics(self, metrics_data: Dict[str, Any]):
        """Add real-time system metrics"""
        with self.lock:
            self.system_metrics_stream.append({
                'timestamp': datetime.now(),
                'data': metrics_data
            })
    
    def _prediction_loop(self):
        """Main prediction generation loop"""
        last_full_analysis = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Generate insights periodically
                if (current_time - last_full_analysis).total_seconds() >= self.insight_refresh_interval:
                    self._generate_all_insights()
                    last_full_analysis = current_time
                
                # Clean up expired insights and recommendations
                self._cleanup_expired_items()
                
                # Sleep for short interval
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                time.sleep(30)
    
    def _generate_all_insights(self):
        """Generate insights for all active sessions and users"""
        try:
            # Generate session-level insights
            active_sessions = list(self.session_performance_stream.keys())
            for session_id in active_sessions[-50:]:  # Limit to recent sessions
                self._generate_session_insights(session_id)
            
            # Generate user-level insights
            active_users = list(self.user_behavior_stream.keys())
            for user_id in active_users[-20:]:  # Limit to recent users
                self._generate_user_insights(user_id)
            
            # Generate system-level insights
            self._generate_system_insights()
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
    
    def _generate_session_insights(self, session_id: str):
        """Generate predictive insights for a specific session"""
        try:
            session_data = self.session_performance_stream.get(session_id)
            if not session_data or len(session_data) < 3:
                return
            
            # Extract features for prediction
            features = self._extract_session_features(session_id, session_data)
            if not features:
                return
            
            # Generate success prediction
            success_insight = self._predict_session_success(session_id, features)
            if success_insight:
                self._store_insight(success_insight)
            
            # Generate performance forecast
            performance_insight = self._forecast_session_performance(session_id, features)
            if performance_insight:
                self._store_insight(performance_insight)
            
            # Generate completion time prediction
            completion_insight = self._predict_completion_time(session_id, features)
            if completion_insight:
                self._store_insight(completion_insight)
            
            # Generate recommendations
            recommendations = self._generate_session_recommendations(session_id, features)
            for rec in recommendations:
                self._store_recommendation(rec)
                
        except Exception as e:
            logger.warning(f"Error generating session insights for {session_id}: {e}")
    
    def _generate_user_insights(self, user_id: str):
        """Generate predictive insights for a specific user"""
        try:
            user_data = self.user_behavior_stream.get(user_id)
            if not user_data or len(user_data) < 5:
                return
            
            # Extract user behavior features
            features = self._extract_user_features(user_id, user_data)
            if not features:
                return
            
            # Generate productivity predictions
            productivity_insight = self._predict_user_productivity(user_id, features)
            if productivity_insight:
                self._store_insight(productivity_insight)
            
            # Generate workflow recommendations
            workflow_recommendations = self._generate_workflow_recommendations(user_id, features)
            for rec in workflow_recommendations:
                self._store_recommendation(rec)
                
        except Exception as e:
            logger.warning(f"Error generating user insights for {user_id}: {e}")
    
    def _generate_system_insights(self):
        """Generate system-level predictive insights"""
        try:
            if len(self.system_metrics_stream) < 10:
                return
            
            # Extract system features
            features = self._extract_system_features()
            if not features:
                return
            
            # Generate resource usage predictions
            resource_insight = self._predict_resource_usage(features)
            if resource_insight:
                self._store_insight(resource_insight)
            
            # Generate system health recommendations
            health_recommendations = self._generate_system_recommendations(features)
            for rec in health_recommendations:
                self._store_recommendation(rec)
                
        except Exception as e:
            logger.warning(f"Error generating system insights: {e}")
    
    # ================================================================
    # PREDICTION METHODS
    # ================================================================
    
    def _predict_session_success(self, session_id: str, features: Dict[str, float]) -> Optional[PredictiveInsight]:
        """Predict session success probability"""
        try:
            if not ML_AVAILABLE:
                return self._fallback_session_success_prediction(session_id, features)
            
            model = self.prediction_models.get('session_success_predictor')
            if not model or not hasattr(model, 'predict_proba'):
                return self._fallback_session_success_prediction(session_id, features)
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, 'session_success_predictor')
            if feature_vector is None:
                return self._fallback_session_success_prediction(session_id, features)
            
            # Make prediction
            success_probability = model.predict_proba([feature_vector])[0]
            confidence = max(success_probability)
            prediction = 'high_success' if success_probability[1] > 0.7 else 'medium_success' if success_probability[1] > 0.4 else 'low_success'
            
            # Generate reasoning
            reasoning = self._generate_success_reasoning(features, success_probability)
            
            # Generate recommendations
            recommendations = self._generate_success_recommendations(prediction, features)
            
            return PredictiveInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='success_prediction',
                target=session_id,
                prediction={
                    'success_level': prediction,
                    'probability': float(success_probability[1]),
                    'confidence': float(confidence)
                },
                confidence=float(confidence),
                reasoning=reasoning,
                recommended_actions=recommendations,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30),
                ml_features=features,
                impact_score=float(success_probability[1])
            )
            
        except Exception as e:
            logger.warning(f"Error in session success prediction: {e}")
            return self._fallback_session_success_prediction(session_id, features)
    
    def _forecast_session_performance(self, session_id: str, features: Dict[str, float]) -> Optional[PredictiveInsight]:
        """Forecast session performance trajectory"""
        try:
            if not ML_AVAILABLE:
                return self._fallback_performance_forecast(session_id, features)
            
            model = self.prediction_models.get('performance_forecaster')
            if not model or not hasattr(model, 'predict'):
                return self._fallback_performance_forecast(session_id, features)
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, 'performance_forecaster')
            if feature_vector is None:
                return self._fallback_performance_forecast(session_id, features)
            
            # Make prediction
            predicted_performance = model.predict([feature_vector])[0]
            
            # Calculate confidence based on model's historical accuracy
            confidence = self._calculate_model_confidence('performance_forecaster', features)
            
            # Determine trend
            current_performance = features.get('current_performance', 0.5)
            trend = 'improving' if predicted_performance > current_performance + 0.1 else 'declining' if predicted_performance < current_performance - 0.1 else 'stable'
            
            # Generate reasoning
            reasoning = self._generate_performance_reasoning(features, predicted_performance, trend)
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(trend, features)
            
            return PredictiveInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='performance_forecast',
                target=session_id,
                prediction={
                    'forecasted_performance': float(predicted_performance),
                    'current_performance': current_performance,
                    'trend': trend,
                    'change_magnitude': float(abs(predicted_performance - current_performance))
                },
                confidence=confidence,
                reasoning=reasoning,
                recommended_actions=recommendations,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=45),
                ml_features=features,
                impact_score=float(abs(predicted_performance - current_performance))
            )
            
        except Exception as e:
            logger.warning(f"Error in performance forecast: {e}")
            return self._fallback_performance_forecast(session_id, features)
    
    def _predict_completion_time(self, session_id: str, features: Dict[str, float]) -> Optional[PredictiveInsight]:
        """Predict session completion time"""
        try:
            if not ML_AVAILABLE:
                return self._fallback_completion_prediction(session_id, features)
            
            model = self.prediction_models.get('completion_time_predictor')
            if not model or not hasattr(model, 'predict'):
                return self._fallback_completion_prediction(session_id, features)
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features, 'completion_time_predictor')
            if feature_vector is None:
                return self._fallback_completion_prediction(session_id, features)
            
            # Make prediction (in minutes)
            predicted_minutes = model.predict([feature_vector])[0]
            predicted_completion = datetime.now() + timedelta(minutes=max(1, predicted_minutes))
            
            # Calculate confidence
            confidence = self._calculate_model_confidence('completion_time_predictor', features)
            
            # Generate reasoning
            reasoning = [
                f"Based on current progress rate: {features.get('progress_rate', 0):.2f}",
                f"Session complexity factor: {features.get('complexity_factor', 1.0):.2f}",
                f"Historical completion patterns considered"
            ]
            
            # Generate recommendations
            recommendations = []
            if predicted_minutes > 120:  # More than 2 hours
                recommendations.append("Consider breaking down into smaller sub-tasks")
                recommendations.append("Schedule regular progress checkpoints")
            elif predicted_minutes < 15:  # Less than 15 minutes
                recommendations.append("Good opportunity to complete quickly")
                recommendations.append("Focus on maintaining current pace")
            
            return PredictiveInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='completion_prediction',
                target=session_id,
                prediction={
                    'estimated_completion_time': predicted_completion.isoformat(),
                    'estimated_minutes_remaining': float(max(1, predicted_minutes)),
                    'completion_category': 'quick' if predicted_minutes < 30 else 'moderate' if predicted_minutes < 120 else 'extended'
                },
                confidence=confidence,
                reasoning=reasoning,
                recommended_actions=recommendations,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=20),
                ml_features=features,
                impact_score=float(min(predicted_minutes / 60, 5.0))  # Cap at 5 hours
            )
            
        except Exception as e:
            logger.warning(f"Error in completion time prediction: {e}")
            return self._fallback_completion_prediction(session_id, features)
    
    def _predict_user_productivity(self, user_id: str, features: Dict[str, float]) -> Optional[PredictiveInsight]:
        """Predict user productivity trends"""
        try:
            # Extract productivity indicators
            current_productivity = features.get('current_productivity', 0.5)
            recent_trend = features.get('productivity_trend', 0.0)
            session_frequency = features.get('session_frequency', 1.0)
            
            # Simple productivity prediction model
            predicted_productivity = current_productivity + (recent_trend * 0.3) + (session_frequency * 0.1)
            predicted_productivity = max(0.0, min(1.0, predicted_productivity))
            
            # Calculate confidence based on data consistency
            confidence = min(0.9, features.get('data_consistency', 0.5) + 0.3)
            
            # Determine productivity level
            if predicted_productivity > 0.8:
                level = 'high'
            elif predicted_productivity > 0.6:
                level = 'moderate'
            else:
                level = 'low'
            
            # Generate reasoning
            reasoning = [
                f"Current productivity level: {current_productivity:.2f}",
                f"Recent trend: {'positive' if recent_trend > 0 else 'negative' if recent_trend < 0 else 'stable'}",
                f"Session frequency factor: {session_frequency:.2f}"
            ]
            
            # Generate recommendations
            recommendations = []
            if level == 'high':
                recommendations.extend([
                    "Maintain current workflow patterns",
                    "Consider taking on more challenging tasks"
                ])
            elif level == 'moderate':
                recommendations.extend([
                    "Look for optimization opportunities",
                    "Consider workflow adjustments"
                ])
            else:
                recommendations.extend([
                    "Focus on removing productivity blockers",
                    "Consider break scheduling optimization",
                    "Review task prioritization"
                ])
            
            return PredictiveInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='productivity_prediction',
                target=user_id,
                prediction={
                    'predicted_productivity': float(predicted_productivity),
                    'productivity_level': level,
                    'trend_direction': 'up' if recent_trend > 0.05 else 'down' if recent_trend < -0.05 else 'stable'
                },
                confidence=confidence,
                reasoning=reasoning,
                recommended_actions=recommendations,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=2),
                ml_features=features,
                impact_score=float(predicted_productivity)
            )
            
        except Exception as e:
            logger.warning(f"Error in user productivity prediction: {e}")
            return None
    
    def _predict_resource_usage(self, features: Dict[str, float]) -> Optional[PredictiveInsight]:
        """Predict system resource usage"""
        try:
            current_usage = features.get('current_resource_usage', 0.5)
            growth_rate = features.get('usage_growth_rate', 0.0)
            load_trend = features.get('load_trend', 0.0)
            
            # Predict resource usage in next hour
            predicted_usage = current_usage + (growth_rate * 0.5) + (load_trend * 0.3)
            predicted_usage = max(0.0, min(1.0, predicted_usage))
            
            # Determine risk level
            if predicted_usage > 0.9:
                risk_level = 'critical'
            elif predicted_usage > 0.75:
                risk_level = 'high'
            elif predicted_usage > 0.5:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
            
            confidence = 0.8  # Reasonable confidence for system metrics
            
            reasoning = [
                f"Current usage: {current_usage:.1%}",
                f"Growth rate: {growth_rate:.3f}",
                f"Load trend: {'increasing' if load_trend > 0 else 'decreasing' if load_trend < 0 else 'stable'}"
            ]
            
            recommendations = []
            if risk_level == 'critical':
                recommendations.extend([
                    "Immediate attention required - resource exhaustion imminent",
                    "Consider scaling up resources",
                    "Implement load balancing"
                ])
            elif risk_level == 'high':
                recommendations.extend([
                    "Monitor closely - approaching resource limits",
                    "Prepare scaling strategy",
                    "Optimize resource-intensive processes"
                ])
            else:
                recommendations.extend([
                    "Resource usage within normal parameters",
                    "Continue monitoring trends"
                ])
            
            return PredictiveInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='resource_prediction',
                target='system',
                prediction={
                    'predicted_usage': float(predicted_usage),
                    'risk_level': risk_level,
                    'time_to_limit': self._estimate_time_to_limit(current_usage, growth_rate)
                },
                confidence=confidence,
                reasoning=reasoning,
                recommended_actions=recommendations,
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30),
                ml_features=features,
                impact_score=float(predicted_usage)
            )
            
        except Exception as e:
            logger.warning(f"Error in resource usage prediction: {e}")
            return None
    
    # ================================================================
    # FEATURE EXTRACTION METHODS
    # ================================================================
    
    def _extract_session_features(self, session_id: str, session_data: deque) -> Optional[Dict[str, float]]:
        """Extract features from session performance data"""
        try:
            if len(session_data) < 3:
                return None
            
            recent_data = list(session_data)[-10:]  # Last 10 data points
            
            # Extract performance metrics
            performance_scores = [d['data'].get('performance_score', 0.5) for d in recent_data]
            operation_counts = [d['data'].get('operation_count', 0) for d in recent_data]
            error_counts = [d['data'].get('error_count', 0) for d in recent_data]
            response_times = [d['data'].get('response_time', 1000) for d in recent_data]
            
            # Calculate features
            features = {
                'current_performance': performance_scores[-1] if performance_scores else 0.5,
                'avg_performance': statistics.mean(performance_scores) if performance_scores else 0.5,
                'performance_trend': self._calculate_trend(performance_scores),
                'operation_rate': statistics.mean(operation_counts) if operation_counts else 0.0,
                'error_rate': sum(error_counts) / max(sum(operation_counts), 1),
                'avg_response_time': statistics.mean(response_times) if response_times else 1000,
                'session_duration': (recent_data[-1]['timestamp'] - recent_data[0]['timestamp']).total_seconds() / 60,
                'data_consistency': self._calculate_consistency(performance_scores),
                'progress_rate': self._calculate_progress_rate(recent_data),
                'complexity_factor': self._estimate_complexity(recent_data)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting session features: {e}")
            return None
    
    def _extract_user_features(self, user_id: str, user_data: deque) -> Optional[Dict[str, float]]:
        """Extract features from user behavior data"""
        try:
            if len(user_data) < 5:
                return None
            
            recent_data = list(user_data)[-20:]  # Last 20 data points
            
            # Extract user behavior metrics
            session_durations = [d['data'].get('session_duration', 60) for d in recent_data]
            productivity_scores = [d['data'].get('productivity_score', 0.5) for d in recent_data]
            task_completion_rates = [d['data'].get('completion_rate', 0.5) for d in recent_data]
            
            features = {
                'current_productivity': productivity_scores[-1] if productivity_scores else 0.5,
                'avg_productivity': statistics.mean(productivity_scores) if productivity_scores else 0.5,
                'productivity_trend': self._calculate_trend(productivity_scores),
                'session_frequency': len(recent_data) / 24,  # Sessions per day
                'avg_session_duration': statistics.mean(session_durations) if session_durations else 60,
                'completion_rate': statistics.mean(task_completion_rates) if task_completion_rates else 0.5,
                'data_consistency': self._calculate_consistency(productivity_scores),
                'activity_pattern': self._analyze_activity_pattern(recent_data)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting user features: {e}")
            return None
    
    def _extract_system_features(self) -> Optional[Dict[str, float]]:
        """Extract features from system metrics"""
        try:
            if len(self.system_metrics_stream) < 10:
                return None
            
            recent_data = list(self.system_metrics_stream)[-30:]  # Last 30 data points
            
            # Extract system metrics
            cpu_usage = [d['data'].get('cpu_usage', 0.5) for d in recent_data]
            memory_usage = [d['data'].get('memory_usage', 0.5) for d in recent_data]
            active_sessions = [d['data'].get('active_sessions', 0) for d in recent_data]
            
            features = {
                'current_resource_usage': max(
                    cpu_usage[-1] if cpu_usage else 0.5,
                    memory_usage[-1] if memory_usage else 0.5
                ),
                'avg_cpu_usage': statistics.mean(cpu_usage) if cpu_usage else 0.5,
                'avg_memory_usage': statistics.mean(memory_usage) if memory_usage else 0.5,
                'usage_growth_rate': self._calculate_trend(cpu_usage + memory_usage),
                'load_trend': self._calculate_trend(active_sessions),
                'system_stability': self._calculate_consistency(cpu_usage + memory_usage),
                'peak_usage': max(cpu_usage + memory_usage) if cpu_usage or memory_usage else 0.5
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting system features: {e}")
            return None
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction and magnitude"""
        if len(values) < 3:
            return 0.0
        
        try:
            # Simple linear trend calculation
            x = list(range(len(values)))
            n = len(values)
            
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            return slope
            
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """Calculate data consistency (inverse of variance)"""
        if len(values) < 2:
            return 0.5
        
        try:
            variance = statistics.variance(values)
            # Convert variance to consistency score (0-1)
            consistency = 1.0 / (1.0 + variance)
            return min(1.0, consistency)
        except statistics.StatisticsError:
            return 0.5
    
    def _calculate_progress_rate(self, data_points: List[Dict]) -> float:
        """Calculate progress rate from data points"""
        if len(data_points) < 2:
            return 0.0
        
        try:
            # Look for progress indicators in the data
            progress_values = []
            for point in data_points:
                progress = point['data'].get('progress', point['data'].get('completion_percentage', 0))
                progress_values.append(float(progress))
            
            if not progress_values:
                return 0.0
            
            # Calculate progress rate (change per minute)
            time_span = (data_points[-1]['timestamp'] - data_points[0]['timestamp']).total_seconds() / 60
            progress_change = progress_values[-1] - progress_values[0]
            
            return progress_change / max(time_span, 1) if time_span > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_complexity(self, data_points: List[Dict]) -> float:
        """Estimate task complexity from data patterns"""
        try:
            # Look for complexity indicators
            error_rates = [point['data'].get('error_count', 0) for point in data_points]
            response_times = [point['data'].get('response_time', 1000) for point in data_points]
            
            # Higher error rates and response times indicate higher complexity
            avg_error_rate = statistics.mean(error_rates) if error_rates else 0
            avg_response_time = statistics.mean(response_times) if response_times else 1000
            
            # Normalize to 0-2 scale (1.0 = normal complexity)
            complexity = 1.0 + (avg_error_rate * 0.5) + (min(avg_response_time / 2000, 1.0))
            return min(2.0, complexity)
            
        except Exception:
            return 1.0
    
    def _analyze_activity_pattern(self, data_points: List[Dict]) -> float:
        """Analyze user activity patterns"""
        try:
            # Extract timestamps and look for patterns
            timestamps = [point['timestamp'] for point in data_points]
            
            # Calculate time intervals
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 60
                intervals.append(interval)
            
            if not intervals:
                return 0.5
            
            # Consistent intervals indicate regular patterns
            consistency = self._calculate_consistency(intervals)
            return consistency
            
        except Exception:
            return 0.5
    
    def _estimate_time_to_limit(self, current_usage: float, growth_rate: float) -> Optional[str]:
        """Estimate time until resource limit is reached"""
        if growth_rate <= 0:
            return None
        
        try:
            remaining_capacity = 1.0 - current_usage
            time_to_limit_hours = remaining_capacity / growth_rate
            
            if time_to_limit_hours < 1:
                return f"{int(time_to_limit_hours * 60)} minutes"
            elif time_to_limit_hours < 24:
                return f"{time_to_limit_hours:.1f} hours"
            else:
                return f"{time_to_limit_hours / 24:.1f} days"
                
        except Exception:
            return None
    
    def _prepare_feature_vector(self, features: Dict[str, float], model_name: str) -> Optional[List[float]]:
        """Prepare feature vector for ML model"""
        try:
            # Define expected features for each model
            feature_sets = {
                'session_success_predictor': ['current_performance', 'avg_performance', 'operation_rate', 'error_rate'],
                'performance_forecaster': ['current_performance', 'performance_trend', 'operation_rate', 'error_rate'],
                'completion_time_predictor': ['progress_rate', 'complexity_factor', 'current_performance'],
                'quality_predictor': ['current_performance', 'error_rate', 'consistency'],
                'resource_usage_predictor': ['current_resource_usage', 'usage_growth_rate', 'load_trend']
            }
            
            expected_features = feature_sets.get(model_name, list(features.keys())[:5])
            
            # Extract feature values
            feature_vector = []
            for feature_name in expected_features:
                value = features.get(feature_name, 0.0)
                feature_vector.append(float(value))
            
            return feature_vector if feature_vector else None
            
        except Exception as e:
            logger.warning(f"Error preparing feature vector: {e}")
            return None
    
    def _calculate_model_confidence(self, model_name: str, features: Dict[str, float]) -> float:
        """Calculate confidence in model prediction"""
        # Simple confidence calculation based on feature quality
        data_quality = features.get('data_consistency', 0.5)
        feature_completeness = len([v for v in features.values() if v is not None]) / len(features)
        
        confidence = (data_quality + feature_completeness) / 2
        return max(0.1, min(0.95, confidence))
    
    # ================================================================
    # FALLBACK PREDICTION METHODS
    # ================================================================
    
    def _fallback_session_success_prediction(self, session_id: str, features: Dict[str, float]) -> Optional[PredictiveInsight]:
        """Fallback session success prediction without ML"""
        try:
            current_performance = features.get('current_performance', 0.5)
            error_rate = features.get('error_rate', 0.0)
            trend = features.get('performance_trend', 0.0)
            
            # Simple heuristic-based prediction
            success_score = current_performance - (error_rate * 0.5) + (trend * 0.3)
            success_score = max(0.0, min(1.0, success_score))
            
            if success_score > 0.7:
                prediction = 'high_success'
            elif success_score > 0.4:
                prediction = 'medium_success'
            else:
                prediction = 'low_success'
            
            return PredictiveInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='success_prediction',
                target=session_id,
                prediction={
                    'success_level': prediction,
                    'probability': float(success_score),
                    'confidence': 0.6
                },
                confidence=0.6,
                reasoning=['Heuristic-based prediction using performance and error metrics'],
                recommended_actions=['Monitor session progress closely'],
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=30),
                ml_features=features
            )
            
        except Exception:
            return None
    
    def _fallback_performance_forecast(self, session_id: str, features: Dict[str, float]) -> Optional[PredictiveInsight]:
        """Fallback performance forecast without ML"""
        try:
            current_performance = features.get('current_performance', 0.5)
            trend = features.get('performance_trend', 0.0)
            
            # Simple trend-based forecast
            predicted_performance = current_performance + (trend * 2.0)  # Extrapolate trend
            predicted_performance = max(0.0, min(1.0, predicted_performance))
            
            trend_direction = 'improving' if trend > 0.05 else 'declining' if trend < -0.05 else 'stable'
            
            return PredictiveInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='performance_forecast',
                target=session_id,
                prediction={
                    'forecasted_performance': float(predicted_performance),
                    'current_performance': current_performance,
                    'trend': trend_direction
                },
                confidence=0.5,
                reasoning=['Simple trend-based forecast'],
                recommended_actions=['Continue monitoring performance trends'],
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=45),
                ml_features=features
            )
            
        except Exception:
            return None
    
    def _fallback_completion_prediction(self, session_id: str, features: Dict[str, float]) -> Optional[PredictiveInsight]:
        """Fallback completion time prediction without ML"""
        try:
            progress_rate = features.get('progress_rate', 0.01)
            complexity_factor = features.get('complexity_factor', 1.0)
            
            # Simple time estimation
            if progress_rate > 0:
                estimated_minutes = 60 * complexity_factor / max(progress_rate, 0.01)
            else:
                estimated_minutes = 90 * complexity_factor  # Default estimate
            
            estimated_minutes = max(5, min(480, estimated_minutes))  # 5 min to 8 hours
            
            return PredictiveInsight(
                insight_id=str(uuid.uuid4()),
                insight_type='completion_prediction',
                target=session_id,
                prediction={
                    'estimated_minutes_remaining': float(estimated_minutes),
                    'completion_category': 'quick' if estimated_minutes < 30 else 'moderate' if estimated_minutes < 120 else 'extended'
                },
                confidence=0.4,
                reasoning=['Heuristic-based estimation using progress rate and complexity'],
                recommended_actions=['Monitor progress and adjust estimates as needed'],
                timestamp=datetime.now(),
                expires_at=datetime.now() + timedelta(minutes=20),
                ml_features=features
            )
            
        except Exception:
            return None
    
    # ================================================================
    # RECOMMENDATION GENERATION
    # ================================================================
    
    def _generate_session_recommendations(self, session_id: str, features: Dict[str, float]) -> List[RealtimeRecommendation]:
        """Generate real-time recommendations for session optimization"""
        recommendations = []
        
        try:
            current_performance = features.get('current_performance', 0.5)
            error_rate = features.get('error_rate', 0.0)
            response_time = features.get('avg_response_time', 1000)
            
            # Performance-based recommendations
            if current_performance < 0.4:
                recommendations.append(RealtimeRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    category='performance',
                    priority='high',
                    title='Low Performance Detected',
                    description='Session performance is below optimal levels',
                    context={'session_id': session_id, 'performance': current_performance},
                    estimated_impact='positive',
                    implementation_effort='low',
                    timestamp=datetime.now(),
                    valid_until=datetime.now() + timedelta(hours=1)
                ))
            
            # Error rate recommendations
            if error_rate > 0.1:
                recommendations.append(RealtimeRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    category='quality',
                    priority='medium',
                    title='High Error Rate Alert',
                    description='Error rate is elevated and may impact session success',
                    context={'session_id': session_id, 'error_rate': error_rate},
                    estimated_impact='positive',
                    implementation_effort='medium',
                    timestamp=datetime.now(),
                    valid_until=datetime.now() + timedelta(hours=2)
                ))
            
            # Response time recommendations
            if response_time > 3000:
                recommendations.append(RealtimeRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    category='performance',
                    priority='medium',
                    title='High Response Time',
                    description='Response times are elevated and may affect user experience',
                    context={'session_id': session_id, 'response_time': response_time},
                    estimated_impact='positive',
                    implementation_effort='medium',
                    timestamp=datetime.now(),
                    valid_until=datetime.now() + timedelta(hours=1)
                ))
            
        except Exception as e:
            logger.warning(f"Error generating session recommendations: {e}")
        
        return recommendations
    
    def _generate_workflow_recommendations(self, user_id: str, features: Dict[str, float]) -> List[RealtimeRecommendation]:
        """Generate workflow optimization recommendations"""
        recommendations = []
        
        try:
            productivity = features.get('current_productivity', 0.5)
            completion_rate = features.get('completion_rate', 0.5)
            
            # Productivity recommendations
            if productivity < 0.6:
                recommendations.append(RealtimeRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    category='workflow',
                    priority='medium',
                    title='Productivity Optimization',
                    description='Consider workflow adjustments to improve productivity',
                    context={'user_id': user_id, 'productivity': productivity},
                    estimated_impact='positive',
                    implementation_effort='low',
                    timestamp=datetime.now(),
                    valid_until=datetime.now() + timedelta(hours=4)
                ))
            
            # Task completion recommendations
            if completion_rate < 0.7:
                recommendations.append(RealtimeRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    category='workflow',
                    priority='low',
                    title='Task Completion Focus',
                    description='Focus on completing current tasks before starting new ones',
                    context={'user_id': user_id, 'completion_rate': completion_rate},
                    estimated_impact='positive',
                    implementation_effort='low',
                    timestamp=datetime.now(),
                    valid_until=datetime.now() + timedelta(hours=6)
                ))
            
        except Exception as e:
            logger.warning(f"Error generating workflow recommendations: {e}")
        
        return recommendations
    
    def _generate_system_recommendations(self, features: Dict[str, float]) -> List[RealtimeRecommendation]:
        """Generate system optimization recommendations"""
        recommendations = []
        
        try:
            resource_usage = features.get('current_resource_usage', 0.5)
            growth_rate = features.get('usage_growth_rate', 0.0)
            
            # Resource usage recommendations
            if resource_usage > 0.8:
                recommendations.append(RealtimeRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    category='resource',
                    priority='high',
                    title='High Resource Usage',
                    description='System resources are approaching limits',
                    context={'resource_usage': resource_usage},
                    estimated_impact='positive',
                    implementation_effort='high',
                    timestamp=datetime.now(),
                    valid_until=datetime.now() + timedelta(hours=1)
                ))
            
            # Growth rate recommendations
            if growth_rate > 0.1:
                recommendations.append(RealtimeRecommendation(
                    recommendation_id=str(uuid.uuid4()),
                    category='resource',
                    priority='medium',
                    title='Resource Growth Monitoring',
                    description='Resource usage is growing rapidly and needs monitoring',
                    context={'growth_rate': growth_rate},
                    estimated_impact='positive',
                    implementation_effort='low',
                    timestamp=datetime.now(),
                    valid_until=datetime.now() + timedelta(hours=2)
                ))
            
        except Exception as e:
            logger.warning(f"Error generating system recommendations: {e}")
        
        return recommendations
    
    # ================================================================
    # REASONING GENERATION
    # ================================================================
    
    def _generate_success_reasoning(self, features: Dict[str, float], success_probability: List[float]) -> List[str]:
        """Generate reasoning for success predictions"""
        reasoning = []
        
        current_performance = features.get('current_performance', 0.5)
        error_rate = features.get('error_rate', 0.0)
        trend = features.get('performance_trend', 0.0)
        
        reasoning.append(f"Current performance score: {current_performance:.2f}")
        
        if error_rate > 0.1:
            reasoning.append(f"Elevated error rate ({error_rate:.1%}) may impact success")
        elif error_rate < 0.05:
            reasoning.append("Low error rate indicates stable execution")
        
        if trend > 0.05:
            reasoning.append("Positive performance trend supports success probability")
        elif trend < -0.05:
            reasoning.append("Declining performance trend may reduce success probability")
        
        return reasoning
    
    def _generate_performance_reasoning(self, features: Dict[str, float], predicted_performance: float, trend: str) -> List[str]:
        """Generate reasoning for performance forecasts"""
        reasoning = []
        
        current_performance = features.get('current_performance', 0.5)
        operation_rate = features.get('operation_rate', 0.0)
        
        reasoning.append(f"Current performance baseline: {current_performance:.2f}")
        reasoning.append(f"Predicted performance trajectory: {trend}")
        
        if operation_rate > 10:
            reasoning.append("High operation rate indicates active usage")
        elif operation_rate < 2:
            reasoning.append("Low operation rate may indicate reduced activity")
        
        performance_change = predicted_performance - current_performance
        if abs(performance_change) > 0.1:
            reasoning.append(f"Significant performance change expected: {performance_change:+.2f}")
        
        return reasoning
    
    def _generate_success_recommendations(self, prediction: str, features: Dict[str, float]) -> List[str]:
        """Generate recommendations based on success prediction"""
        recommendations = []
        
        error_rate = features.get('error_rate', 0.0)
        current_performance = features.get('current_performance', 0.5)
        
        if prediction == 'low_success':
            recommendations.extend([
                "Consider breaking down complex tasks into smaller steps",
                "Review and address any error conditions",
                "Monitor progress more frequently"
            ])
        elif prediction == 'medium_success':
            recommendations.extend([
                "Maintain current approach with minor optimizations",
                "Address any emerging error patterns"
            ])
        else:  # high_success
            recommendations.extend([
                "Current approach is working well",
                "Consider documenting successful patterns for future use"
            ])
        
        if error_rate > 0.1:
            recommendations.append("Focus on error reduction to improve success probability")
        
        if current_performance < 0.5:
            recommendations.append("Look for opportunities to optimize performance")
        
        return recommendations
    
    def _generate_performance_recommendations(self, trend: str, features: Dict[str, float]) -> List[str]:
        """Generate recommendations based on performance forecast"""
        recommendations = []
        
        if trend == 'declining':
            recommendations.extend([
                "Investigate factors causing performance decline",
                "Consider workflow adjustments",
                "Monitor system resources"
            ])
        elif trend == 'improving':
            recommendations.extend([
                "Continue current optimization strategies",
                "Document successful performance improvements"
            ])
        else:  # stable
            recommendations.extend([
                "Performance is stable - maintain current approach",
                "Look for opportunities for gradual improvement"
            ])
        
        response_time = features.get('avg_response_time', 1000)
        if response_time > 2000:
            recommendations.append("Investigate response time optimization opportunities")
        
        return recommendations
    
    # ================================================================
    # UTILITY METHODS
    # ================================================================
    
    def _is_significant_change(self, session_id: str, performance_data: Dict[str, Any]) -> bool:
        """Check if performance change is significant enough to trigger predictions"""
        session_data = self.session_performance_stream.get(session_id)
        if not session_data or len(session_data) < 2:
            return True  # Always trigger for new sessions
        
        try:
            current_performance = performance_data.get('performance_score', 0.5)
            previous_data = list(session_data)[-1]
            previous_performance = previous_data['data'].get('performance_score', 0.5)
            
            # Trigger if performance change > 10% or new error
            performance_change = abs(current_performance - previous_performance)
            new_error = performance_data.get('error_count', 0) > previous_data['data'].get('error_count', 0)
            
            return performance_change > 0.1 or new_error
            
        except Exception:
            return True  # Default to triggering predictions
    
    def _store_insight(self, insight: PredictiveInsight):
        """Store generated insight"""
        with self.lock:
            self.active_insights[insight.insight_id] = insight
            self.insight_history.append(insight)
    
    def _store_recommendation(self, recommendation: RealtimeRecommendation):
        """Store generated recommendation"""
        with self.lock:
            self.recommendations[recommendation.recommendation_id] = recommendation
    
    def _cleanup_expired_items(self):
        """Clean up expired insights and recommendations"""
        try:
            current_time = datetime.now()
            
            # Clean up expired insights
            expired_insights = [
                insight_id for insight_id, insight in self.active_insights.items()
                if insight.expires_at and insight.expires_at < current_time
            ]
            
            for insight_id in expired_insights:
                self.active_insights.pop(insight_id, None)
            
            # Clean up expired recommendations
            expired_recommendations = [
                rec_id for rec_id, rec in self.recommendations.items()
                if rec.valid_until and rec.valid_until < current_time
            ]
            
            for rec_id in expired_recommendations:
                self.recommendations.pop(rec_id, None)
            
            # Also clean up old recommendations beyond TTL
            ttl_cutoff = current_time - timedelta(hours=self.recommendation_ttl_hours)
            old_recommendations = [
                rec_id for rec_id, rec in self.recommendations.items()
                if rec.timestamp < ttl_cutoff
            ]
            
            for rec_id in old_recommendations:
                self.recommendations.pop(rec_id, None)
                
        except Exception as e:
            logger.warning(f"Error cleaning up expired items: {e}")
    
    # ================================================================
    # PUBLIC API METHODS
    # ================================================================
    
    def get_active_insights(self, target: Optional[str] = None, 
                          insight_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently active predictive insights"""
        with self.lock:
            insights = []
            
            for insight in self.active_insights.values():
                if target and insight.target != target:
                    continue
                if insight_type and insight.insight_type != insight_type:
                    continue
                
                insights.append(asdict(insight))
            
            return sorted(insights, key=lambda x: x['timestamp'], reverse=True)
    
    def get_active_recommendations(self, category: Optional[str] = None,
                                 priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get currently active recommendations"""
        with self.lock:
            recommendations = []
            
            for rec in self.recommendations.values():
                if category and rec.category != category:
                    continue
                if priority and rec.priority != priority:
                    continue
                
                recommendations.append(asdict(rec))
            
            return sorted(recommendations, key=lambda x: x['timestamp'], reverse=True)
    
    def get_insight_summary(self) -> Dict[str, Any]:
        """Get summary of predictive insights status"""
        with self.lock:
            return {
                'active_insights': len(self.active_insights),
                'active_recommendations': len(self.recommendations),
                'insights_by_type': self._count_by_attribute('insight_type', self.active_insights),
                'recommendations_by_category': self._count_by_attribute('category', self.recommendations),
                'recommendations_by_priority': self._count_by_attribute('priority', self.recommendations),
                'prediction_engine_status': 'running' if self.is_running else 'stopped',
                'ml_available': ML_AVAILABLE
            }
    
    def _count_by_attribute(self, attribute: str, items: Dict) -> Dict[str, int]:
        """Count items by attribute"""
        counts = defaultdict(int)
        for item in items.values():
            attr_value = getattr(item, attribute, 'unknown')
            counts[attr_value] += 1
        return dict(counts)
    
    def get_prediction_status(self) -> Dict[str, Any]:
        """Get prediction engine status"""
        return {
            'running': self.is_running,
            'ml_available': ML_AVAILABLE,
            'registered_models': len(self.prediction_models),
            'active_data_streams': {
                'session_performance': len(self.session_performance_stream),
                'user_behavior': len(self.user_behavior_stream),
                'system_metrics': len(self.system_metrics_stream)
            },
            'insight_refresh_interval': self.insight_refresh_interval,
            'confidence_threshold': self.confidence_threshold
        }