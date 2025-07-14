#!/usr/bin/env python3
"""
Phase 6: ML Predictive Analytics Engine
Advanced predictive capabilities for session recommendations and insights
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import math

# ML imports for predictive analytics
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
    import scipy.stats as stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available - using fallback predictive analytics")

logger = logging.getLogger(__name__)

class MLPredictiveAnalytics:
    """
    Machine Learning Predictive Analytics Engine
    Provides advanced predictive capabilities for session analysis
    """
    
    def __init__(self, db_manager, session_manager):
        self.db_manager = db_manager
        self.session_manager = session_manager
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Initialize predictive models
        self._initialize_predictive_models()
    
    def _initialize_predictive_models(self):
        """Initialize predictive models and scalers"""
        if SKLEARN_AVAILABLE:
            self.models = {
                'session_success_predictor': RandomForestClassifier(n_estimators=50, random_state=42),
                'performance_predictor': RandomForestRegressor(n_estimators=50, random_state=42),
                'activity_forecaster': LinearRegression(),
                'anomaly_predictor': LogisticRegression(random_state=42)
            }
            
            self.scalers = {
                'feature_scaler': StandardScaler(),
                'target_scaler': StandardScaler()
            }
            
            logger.info("✅ ML Predictive Analytics Engine initialized with scikit-learn")
        else:
            logger.warning("⚠️ ML Predictive Analytics Engine initialized with fallback implementations")
    
    # ================================================================
    # SESSION SUCCESS PREDICTION
    # ================================================================
    
    def predict_session_success(self, session_features: Dict[str, Any], user_history: List) -> Dict[str, Any]:
        """
        Predict likelihood of session success based on features and user history
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_success_prediction(session_features, user_history)
        
        try:
            # Extract features for prediction
            feature_vector = self._extract_success_features(session_features, user_history)
            
            # Train model if not already trained
            if not hasattr(self.models['session_success_predictor'], 'feature_importances_'):
                training_success = self._train_success_predictor(user_history)
                if not training_success:
                    return self._fallback_success_prediction(session_features, user_history)
            
            # Make prediction
            success_probability = self.models['session_success_predictor'].predict_proba([feature_vector])[0]
            
            # Get feature importance
            feature_importance = self._get_feature_importance('session_success_predictor', feature_vector)
            
            # Generate recommendations
            recommendations = self._generate_success_recommendations(feature_vector, feature_importance)
            
            return {
                "success_probability": {
                    "low_success": float(success_probability[0]),
                    "high_success": float(success_probability[1])
                },
                "predicted_outcome": "success" if success_probability[1] > 0.6 else "needs_attention",
                "confidence": float(max(success_probability)),
                "key_factors": feature_importance[:3],
                "recommendations": recommendations,
                "prediction_metadata": {
                    "model_type": "random_forest_classifier",
                    "training_sessions": len(user_history),
                    "predicted_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Session success prediction failed: {e}")
            return self._fallback_success_prediction(session_features, user_history)
    
    def _extract_success_features(self, session_features: Dict[str, Any], user_history: List) -> List[float]:
        """Extract features for success prediction"""
        features = []
        
        # Current session features
        features.extend([
            session_features.get('expected_duration', 60),  # minutes
            session_features.get('complexity_score', 0.5),
            session_features.get('priority_numeric', 2),  # 1-4 scale
            session_features.get('has_clear_objective', 0),  # 0 or 1
            len(session_features.get('project_context', '')),
            session_features.get('estimated_operations', 10)
        ])
        
        # User history features
        if user_history:
            recent_sessions = sorted(user_history, key=lambda x: x.created_at, reverse=True)[:10]
            
            # Historical success rate
            successful_sessions = [s for s in recent_sessions if s.performance_score and s.performance_score > 0.7]
            success_rate = len(successful_sessions) / len(recent_sessions)
            features.append(success_rate)
            
            # Average performance
            avg_performance = np.mean([s.performance_score or 0.5 for s in recent_sessions])
            features.append(avg_performance)
            
            # Session frequency (sessions per week)
            if len(recent_sessions) > 1:
                time_span = (recent_sessions[0].created_at - recent_sessions[-1].created_at).total_seconds() / (7 * 24 * 3600)
                session_frequency = len(recent_sessions) / max(time_span, 1)
            else:
                session_frequency = 1
            features.append(session_frequency)
            
            # Average session complexity
            avg_complexity = np.mean([s.total_operations or 5 for s in recent_sessions])
            features.append(avg_complexity)
        else:
            # Default values for new users
            features.extend([0.5, 0.5, 1.0, 5.0])
        
        return features
    
    def _train_success_predictor(self, user_history: List) -> bool:
        """Train the session success prediction model"""
        if len(user_history) < 10:
            logger.warning("⚠️ Insufficient history for training success predictor")
            return False
        
        try:
            # Prepare training data
            X_train = []
            y_train = []
            
            for session in user_history:
                # Extract features (simplified for training)
                session_features = {
                    'expected_duration': 60,
                    'complexity_score': min(1.0, (session.total_operations or 5) / 20),
                    'priority_numeric': {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(session.priority.value, 2),
                    'has_clear_objective': 1 if session.session_name and len(session.session_name) > 10 else 0,
                    'project_context': session.project_context or '',
                    'estimated_operations': session.total_operations or 5
                }
                
                features = self._extract_success_features(session_features, user_history)
                X_train.append(features)
                
                # Define success (1) vs failure (0)
                success = 1 if (session.performance_score or 0) > 0.6 else 0
                y_train.append(success)
            
            if len(set(y_train)) < 2:
                logger.warning("⚠️ No variance in success outcomes for training")
                return False
            
            # Train model
            X_train_array = np.array(X_train)
            self.scalers['feature_scaler'].fit(X_train_array)
            X_train_scaled = self.scalers['feature_scaler'].transform(X_train_array)
            
            self.models['session_success_predictor'].fit(X_train_scaled, y_train)
            
            # Evaluate model
            if len(X_train) > 5:
                scores = cross_val_score(self.models['session_success_predictor'], X_train_scaled, y_train, cv=3)
                logger.info(f"✅ Success predictor trained with CV score: {np.mean(scores):.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to train success predictor: {e}")
            return False
    
    def _get_feature_importance(self, model_name: str, feature_vector: List[float]) -> List[Dict[str, Any]]:
        """Get feature importance for predictions"""
        feature_names = [
            'expected_duration', 'complexity_score', 'priority', 'has_clear_objective',
            'context_length', 'estimated_operations', 'historical_success_rate',
            'avg_performance', 'session_frequency', 'avg_complexity'
        ]
        
        try:
            model = self.models[model_name]
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                feature_importance = []
                for i, (name, importance, value) in enumerate(zip(feature_names, importances, feature_vector)):
                    feature_importance.append({
                        'feature': name,
                        'importance': float(importance),
                        'value': float(value),
                        'impact': 'positive' if importance > 0.1 else 'neutral'
                    })
                
                # Sort by importance
                feature_importance.sort(key=lambda x: x['importance'], reverse=True)
                return feature_importance
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to get feature importance: {e}")
        
        return []
    
    def _generate_success_recommendations(self, feature_vector: List[float], feature_importance: List[Dict]) -> List[str]:
        """Generate recommendations to improve session success"""
        recommendations = []
        
        if not feature_importance:
            return ["Focus on clear objectives and consistent session practices"]
        
        # Analyze top factors
        for feature in feature_importance[:3]:
            feature_name = feature['feature']
            value = feature['value']
            importance = feature['importance']
            
            if importance > 0.1:  # Significant factor
                if feature_name == 'complexity_score' and value > 0.8:
                    recommendations.append("Consider breaking down complex tasks into smaller sessions")
                elif feature_name == 'has_clear_objective' and value < 0.5:
                    recommendations.append("Define clear objectives before starting the session")
                elif feature_name == 'historical_success_rate' and value < 0.5:
                    recommendations.append("Review and optimize your session workflow patterns")
                elif feature_name == 'session_frequency' and value > 5:
                    recommendations.append("Consider consolidating frequent short sessions")
        
        if not recommendations:
            recommendations.append("Continue with current session practices - metrics look good")
        
        return recommendations
    
    # ================================================================
    # PERFORMANCE PREDICTION
    # ================================================================
    
    def predict_session_performance(self, session_id: str, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict final session performance based on current progress
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_performance_prediction(current_metrics)
        
        try:
            # Get session details
            session = self.session_manager.get_session(session_id)
            if not session:
                return {"error": "Session not found"}
            
            # Extract performance features
            feature_vector = self._extract_performance_features(session, current_metrics)
            
            # Train model if needed
            user_sessions = self.session_manager.list_user_sessions(session.user_id, session.realm_id, limit=100)
            if not hasattr(self.models['performance_predictor'], 'feature_importances_'):
                training_success = self._train_performance_predictor(user_sessions)
                if not training_success:
                    return self._fallback_performance_prediction(current_metrics)
            
            # Make prediction
            predicted_performance = self.models['performance_predictor'].predict([feature_vector])[0]
            
            # Calculate confidence interval
            confidence_interval = self._calculate_prediction_confidence(
                feature_vector, 'performance_predictor'
            )
            
            # Generate performance improvement suggestions
            improvement_suggestions = self._generate_performance_improvements(
                feature_vector, predicted_performance
            )
            
            return {
                "predicted_final_performance": float(max(0, min(1, predicted_performance))),
                "current_performance": current_metrics.get('current_performance', 0.5),
                "performance_trend": self._calculate_performance_trend(predicted_performance, current_metrics),
                "confidence_interval": confidence_interval,
                "improvement_potential": float(max(0, 1 - predicted_performance)),
                "suggestions": improvement_suggestions,
                "prediction_metadata": {
                    "model_type": "random_forest_regressor",
                    "feature_count": len(feature_vector),
                    "predicted_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Performance prediction failed: {e}")
            return self._fallback_performance_prediction(current_metrics)
    
    def _extract_performance_features(self, session, current_metrics: Dict[str, Any]) -> List[float]:
        """Extract features for performance prediction"""
        features = []
        
        # Current session state
        features.extend([
            current_metrics.get('current_performance', 0.5),
            current_metrics.get('entries_so_far', 0),
            current_metrics.get('operations_so_far', 0),
            current_metrics.get('session_duration_minutes', 30),
            current_metrics.get('error_rate', 0.0),
            current_metrics.get('user_engagement_score', 0.7)
        ])
        
        # Session characteristics
        features.extend([
            len(session.session_name or ''),
            len(session.project_context or ''),
            {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}.get(session.priority.value, 2),
            1 if session.session_state.value == 'active' else 0
        ])
        
        # Time-based features
        if session.created_at:
            age_hours = (datetime.now() - session.created_at).total_seconds() / 3600
            features.append(min(168, age_hours))  # Cap at 1 week
            
            # Time of day effect
            hour_of_day = session.created_at.hour
            features.extend([
                1 if 9 <= hour_of_day <= 17 else 0,  # Business hours
                1 if hour_of_day >= 18 or hour_of_day <= 8 else 0  # After hours
            ])
        else:
            features.extend([0, 0, 0])
        
        return features
    
    def _train_performance_predictor(self, sessions: List) -> bool:
        """Train the performance prediction model"""
        if len(sessions) < 15:
            logger.warning("⚠️ Insufficient sessions for training performance predictor")
            return False
        
        try:
            # Prepare training data
            X_train = []
            y_train = []
            
            for session in sessions:
                if session.performance_score is not None:
                    # Simulate current metrics (in practice, this would come from session history)
                    current_metrics = {
                        'current_performance': session.performance_score * 0.8,  # Simulate partial progress
                        'entries_so_far': (session.total_entries or 0) * 0.7,
                        'operations_so_far': (session.total_operations or 0) * 0.7,
                        'session_duration_minutes': 45,
                        'error_rate': 0.05 if session.performance_score < 0.5 else 0.01,
                        'user_engagement_score': min(1.0, session.performance_score + 0.1)
                    }
                    
                    features = self._extract_performance_features(session, current_metrics)
                    X_train.append(features)
                    y_train.append(session.performance_score)
            
            if len(X_train) < 10:
                logger.warning("⚠️ Insufficient training data for performance predictor")
                return False
            
            # Train model
            X_train_array = np.array(X_train)
            self.scalers['feature_scaler'].fit(X_train_array)
            X_train_scaled = self.scalers['feature_scaler'].transform(X_train_array)
            
            self.models['performance_predictor'].fit(X_train_scaled, y_train)
            
            # Evaluate model
            if len(X_train) > 5:
                scores = cross_val_score(self.models['performance_predictor'], X_train_scaled, y_train, cv=3)
                logger.info(f"✅ Performance predictor trained with CV score: {np.mean(scores):.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to train performance predictor: {e}")
            return False
    
    def _calculate_prediction_confidence(self, feature_vector: List[float], model_name: str) -> Dict[str, float]:
        """Calculate confidence interval for predictions"""
        try:
            # Simple confidence calculation based on model variance
            # In practice, you might use more sophisticated uncertainty quantification
            base_confidence = 0.7
            
            # Adjust confidence based on feature completeness
            feature_completeness = sum(1 for f in feature_vector if f != 0) / len(feature_vector)
            adjusted_confidence = base_confidence * feature_completeness
            
            margin = 0.1 * (1 - adjusted_confidence)
            
            return {
                "confidence": float(adjusted_confidence),
                "margin_of_error": float(margin),
                "lower_bound": float(max(0, adjusted_confidence - margin)),
                "upper_bound": float(min(1, adjusted_confidence + margin))
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to calculate prediction confidence: {e}")
            return {"confidence": 0.5, "margin_of_error": 0.2}
    
    def _calculate_performance_trend(self, predicted_performance: float, current_metrics: Dict[str, Any]) -> str:
        """Calculate performance trend direction"""
        current_perf = current_metrics.get('current_performance', 0.5)
        
        if predicted_performance > current_perf + 0.1:
            return "improving"
        elif predicted_performance < current_perf - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _generate_performance_improvements(self, feature_vector: List[float], predicted_performance: float) -> List[str]:
        """Generate suggestions for performance improvement"""
        suggestions = []
        
        # Analyze feature values for improvement opportunities
        current_perf, entries, operations, duration, error_rate, engagement = feature_vector[:6]
        
        if predicted_performance < 0.6:
            if error_rate > 0.05:
                suggestions.append("Focus on reducing errors to improve overall performance")
            if engagement < 0.6:
                suggestions.append("Consider taking breaks to maintain engagement")
            if duration > 120:
                suggestions.append("Break long sessions into shorter, focused segments")
        
        if operations > entries * 3:
            suggestions.append("Optimize operations per entry ratio for better efficiency")
        
        if not suggestions:
            suggestions.append("Current session is on track for good performance")
        
        return suggestions
    
    # ================================================================
    # ACTIVITY FORECASTING
    # ================================================================
    
    def forecast_user_activity(self, user_id: str, forecast_days: int = 7) -> Dict[str, Any]:
        """
        Forecast user activity patterns for the next N days
        """
        if not SKLEARN_AVAILABLE:
            return self._fallback_activity_forecast(user_id, forecast_days)
        
        try:
            # Get user session history
            user_sessions = self.session_manager.list_user_sessions(user_id, 'PROJECT', limit=500)
            
            if len(user_sessions) < 10:
                return {"forecast": "insufficient_history", "days": forecast_days}
            
            # Prepare time series data
            activity_data = self._prepare_activity_timeseries(user_sessions)
            
            if len(activity_data) < 7:
                return {"forecast": "insufficient_timeseries_data", "days": forecast_days}
            
            # Train forecasting model
            X_train, y_train = self._prepare_forecasting_features(activity_data)
            
            if len(X_train) < 5:
                return {"forecast": "insufficient_training_data", "days": forecast_days}
            
            # Fit forecasting model
            self.models['activity_forecaster'].fit(X_train, y_train)
            
            # Generate forecasts
            forecasts = []
            base_date = datetime.now().date()
            
            for day_offset in range(forecast_days):
                forecast_date = base_date + timedelta(days=day_offset + 1)
                
                # Create features for forecast day
                forecast_features = self._create_forecast_features(forecast_date, activity_data)
                predicted_activity = self.models['activity_forecaster'].predict([forecast_features])[0]
                
                forecasts.append({
                    "date": forecast_date.isoformat(),
                    "predicted_sessions": max(0, int(round(predicted_activity))),
                    "confidence": self._calculate_forecast_confidence(day_offset, len(activity_data))
                })
            
            # Calculate forecast summary
            total_predicted = sum(f['predicted_sessions'] for f in forecasts)
            avg_daily = total_predicted / forecast_days
            
            return {
                "user_id": user_id,
                "forecast_period_days": forecast_days,
                "daily_forecasts": forecasts,
                "summary": {
                    "total_predicted_sessions": total_predicted,
                    "average_daily_sessions": avg_daily,
                    "peak_activity_day": max(forecasts, key=lambda x: x['predicted_sessions'])['date'],
                    "activity_trend": self._analyze_forecast_trend(forecasts)
                },
                "model_metadata": {
                    "training_days": len(activity_data),
                    "historical_sessions": len(user_sessions),
                    "model_type": "linear_regression",
                    "forecast_generated_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Activity forecasting failed: {e}")
            return self._fallback_activity_forecast(user_id, forecast_days)
    
    def _prepare_activity_timeseries(self, sessions: List) -> List[Dict[str, Any]]:
        """Prepare time series data from session history"""
        # Group sessions by date
        daily_activity = defaultdict(list)
        
        for session in sessions:
            if session.created_at:
                date_key = session.created_at.date()
                daily_activity[date_key].append(session)
        
        # Convert to time series
        activity_data = []
        sorted_dates = sorted(daily_activity.keys())
        
        for date in sorted_dates:
            day_sessions = daily_activity[date]
            
            activity_data.append({
                'date': date,
                'session_count': len(day_sessions),
                'total_operations': sum(s.total_operations or 0 for s in day_sessions),
                'avg_performance': np.mean([s.performance_score or 0.5 for s in day_sessions]),
                'weekday': date.weekday(),  # 0=Monday, 6=Sunday
                'is_weekend': date.weekday() >= 5
            })
        
        return activity_data
    
    def _prepare_forecasting_features(self, activity_data: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for forecasting model"""
        X = []
        y = []
        
        # Use sliding window for features
        window_size = min(7, len(activity_data) // 2)
        
        for i in range(window_size, len(activity_data)):
            # Features: last N days of activity
            features = []
            for j in range(window_size):
                day_data = activity_data[i - window_size + j]
                features.extend([
                    day_data['session_count'],
                    day_data['weekday'],
                    day_data['is_weekend']
                ])
            
            X.append(features)
            y.append(activity_data[i]['session_count'])
        
        return np.array(X), np.array(y)
    
    def _create_forecast_features(self, forecast_date, activity_data: List[Dict]) -> List[float]:
        """Create features for a forecast date"""
        # Use recent activity patterns
        recent_data = activity_data[-7:]  # Last 7 days
        
        features = []
        
        # Recent activity features
        for day_data in recent_data:
            features.extend([
                day_data['session_count'],
                day_data['weekday'],
                day_data['is_weekend']
            ])
        
        # Pad if insufficient recent data
        while len(features) < 21:  # 7 days * 3 features
            features.extend([0, 0, 0])
        
        return features[:21]  # Ensure consistent feature size
    
    def _calculate_forecast_confidence(self, day_offset: int, training_days: int) -> float:
        """Calculate confidence for forecast based on distance and training data"""
        # Confidence decreases with forecast distance and increases with training data
        distance_penalty = 0.9 ** day_offset
        data_boost = min(1.0, training_days / 30)
        
        return float(0.5 * distance_penalty * data_boost + 0.3)
    
    def _analyze_forecast_trend(self, forecasts: List[Dict]) -> str:
        """Analyze the overall trend in forecast"""
        values = [f['predicted_sessions'] for f in forecasts]
        
        if len(values) < 3:
            return "insufficient_data"
        
        # Simple trend analysis
        first_half = np.mean(values[:len(values)//2])
        second_half = np.mean(values[len(values)//2:])
        
        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    # ================================================================
    # RECOMMENDATION ENGINE
    # ================================================================
    
    def generate_intelligent_recommendations(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate intelligent recommendations using multiple ML models
        """
        try:
            # Get user context
            user_sessions = self.session_manager.list_user_sessions(user_id, 'PROJECT', limit=100)
            current_session = None
            
            if session_id:
                current_session = self.session_manager.get_session(session_id)
            
            recommendations = {
                "user_id": user_id,
                "current_session_id": session_id,
                "recommendation_categories": []
            }
            
            # Session optimization recommendations
            if current_session:
                session_recs = self._generate_session_optimization_recommendations(current_session, user_sessions)
                recommendations["recommendation_categories"].append({
                    "category": "session_optimization",
                    "recommendations": session_recs
                })
            
            # Workflow pattern recommendations
            workflow_recs = self._generate_workflow_recommendations(user_sessions)
            recommendations["recommendation_categories"].append({
                "category": "workflow_patterns",
                "recommendations": workflow_recs
            })
            
            # Time management recommendations
            time_recs = self._generate_time_management_recommendations(user_sessions)
            recommendations["recommendation_categories"].append({
                "category": "time_management",
                "recommendations": time_recs
            })
            
            # Performance improvement recommendations
            perf_recs = self._generate_performance_recommendations(user_sessions)
            recommendations["recommendation_categories"].append({
                "category": "performance_improvement",
                "recommendations": perf_recs
            })
            
            # Predictive recommendations
            if SKLEARN_AVAILABLE:
                predictive_recs = self._generate_predictive_recommendations(user_id, user_sessions)
                recommendations["recommendation_categories"].append({
                    "category": "predictive_insights",
                    "recommendations": predictive_recs
                })
            
            return {
                "success": True,
                "recommendations": recommendations,
                "generated_at": datetime.now().isoformat(),
                "total_categories": len(recommendations["recommendation_categories"])
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to generate intelligent recommendations: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_session_optimization_recommendations(self, session, user_sessions: List) -> List[Dict[str, Any]]:
        """Generate session-specific optimization recommendations"""
        recommendations = []
        
        # Analyze current session characteristics
        current_perf = session.performance_score or 0.5
        current_ops = session.total_operations or 0
        current_entries = session.total_entries or 0
        
        # Compare with user's historical performance
        historical_perfs = [s.performance_score for s in user_sessions if s.performance_score]
        if historical_perfs:
            avg_perf = np.mean(historical_perfs)
            
            if current_perf < avg_perf - 0.2:
                recommendations.append({
                    "type": "performance_alert",
                    "priority": "high",
                    "message": "Current session performance below your average",
                    "suggestion": "Consider reviewing session objectives or taking a break",
                    "confidence": 0.8
                })
        
        # Operations efficiency
        if current_entries > 0 and current_ops / current_entries > 5:
            recommendations.append({
                "type": "efficiency",
                "priority": "medium",
                "message": "High operations-to-entries ratio detected",
                "suggestion": "Focus on consolidating operations for better efficiency",
                "confidence": 0.7
            })
        
        return recommendations
    
    def _generate_workflow_recommendations(self, user_sessions: List) -> List[Dict[str, Any]]:
        """Generate workflow pattern recommendations"""
        recommendations = []
        
        if len(user_sessions) < 5:
            recommendations.append({
                "type": "workflow_analysis",
                "priority": "low",
                "message": "Building session history for better workflow insights",
                "suggestion": "Continue using the system to unlock workflow recommendations",
                "confidence": 0.6
            })
            return recommendations
        
        # Analyze session patterns
        session_durations = []
        for session in user_sessions:
            if session.created_at and session.last_activity:
                duration = (session.last_activity - session.created_at).total_seconds() / 3600
                session_durations.append(duration)
        
        if session_durations:
            avg_duration = np.mean(session_durations)
            
            if avg_duration > 4:  # Long sessions
                recommendations.append({
                    "type": "session_length",
                    "priority": "medium",
                    "message": "Your sessions tend to be quite long",
                    "suggestion": "Consider breaking work into shorter, focused sessions",
                    "confidence": 0.75
                })
            elif avg_duration < 0.5:  # Very short sessions
                recommendations.append({
                    "type": "session_length",
                    "priority": "medium",
                    "message": "Your sessions are typically very short",
                    "suggestion": "Consider consolidating related tasks into longer sessions",
                    "confidence": 0.75
                })
        
        return recommendations
    
    def _generate_time_management_recommendations(self, user_sessions: List) -> List[Dict[str, Any]]:
        """Generate time management recommendations"""
        recommendations = []
        
        # Analyze timing patterns
        creation_hours = []
        for session in user_sessions:
            if session.created_at:
                creation_hours.append(session.created_at.hour)
        
        if creation_hours:
            peak_hour = Counter(creation_hours).most_common(1)[0][0]
            
            if 9 <= peak_hour <= 11:
                recommendations.append({
                    "type": "timing_optimization",
                    "priority": "low",
                    "message": "You're most active in morning hours",
                    "suggestion": "Consider scheduling complex tasks during your peak morning hours",
                    "confidence": 0.6
                })
            elif 14 <= peak_hour <= 16:
                recommendations.append({
                    "type": "timing_optimization",
                    "priority": "low",
                    "message": "You're most active in afternoon hours",
                    "suggestion": "Leverage your afternoon productivity for important tasks",
                    "confidence": 0.6
                })
        
        return recommendations
    
    def _generate_performance_recommendations(self, user_sessions: List) -> List[Dict[str, Any]]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Analyze performance trends
        recent_sessions = sorted(user_sessions, key=lambda x: x.created_at, reverse=True)[:10]
        performance_scores = [s.performance_score for s in recent_sessions if s.performance_score]
        
        if len(performance_scores) >= 3:
            recent_avg = np.mean(performance_scores[:3])
            overall_avg = np.mean(performance_scores)
            
            if recent_avg < overall_avg - 0.1:
                recommendations.append({
                    "type": "performance_decline",
                    "priority": "high",
                    "message": "Recent performance below your usual standards",
                    "suggestion": "Review recent workflow changes or consider adjusting session complexity",
                    "confidence": 0.8
                })
            elif recent_avg > overall_avg + 0.1:
                recommendations.append({
                    "type": "performance_improvement",
                    "priority": "low",
                    "message": "Your recent performance has been excellent",
                    "suggestion": "Continue with current practices and consider documenting successful patterns",
                    "confidence": 0.8
                })
        
        return recommendations
    
    def _generate_predictive_recommendations(self, user_id: str, user_sessions: List) -> List[Dict[str, Any]]:
        """Generate recommendations based on predictive models"""
        recommendations = []
        
        try:
            # Activity forecast recommendations
            forecast = self.forecast_user_activity(user_id, 7)
            
            if forecast.get('summary'):
                avg_daily = forecast['summary']['average_daily_sessions']
                
                if avg_daily > 5:
                    recommendations.append({
                        "type": "activity_forecast",
                        "priority": "medium",
                        "message": f"Predicting high activity next week ({avg_daily:.1f} sessions/day)",
                        "suggestion": "Plan for session management and ensure adequate breaks",
                        "confidence": 0.7
                    })
                elif avg_daily < 1:
                    recommendations.append({
                        "type": "activity_forecast",
                        "priority": "low",
                        "message": f"Predicting low activity next week ({avg_daily:.1f} sessions/day)",
                        "suggestion": "Consider scheduling focused work sessions to maintain momentum",
                        "confidence": 0.7
                    })
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate predictive recommendations: {e}")
        
        return recommendations
    
    # ================================================================
    # FALLBACK IMPLEMENTATIONS
    # ================================================================
    
    def _fallback_success_prediction(self, session_features: Dict[str, Any], user_history: List) -> Dict[str, Any]:
        """Fallback success prediction when ML is not available"""
        # Simple heuristic-based prediction
        complexity = session_features.get('complexity_score', 0.5)
        has_objective = session_features.get('has_clear_objective', 0)
        
        if user_history:
            historical_success = len([s for s in user_history if (s.performance_score or 0) > 0.6]) / len(user_history)
        else:
            historical_success = 0.5
        
        # Simple success probability
        success_prob = (1 - complexity) * 0.3 + has_objective * 0.3 + historical_success * 0.4
        
        return {
            "success_probability": {
                "low_success": 1 - success_prob,
                "high_success": success_prob
            },
            "predicted_outcome": "success" if success_prob > 0.6 else "needs_attention",
            "confidence": 0.5,
            "key_factors": [{"feature": "heuristic_based", "importance": 1.0}],
            "recommendations": ["Install scikit-learn for advanced ML predictions"],
            "prediction_metadata": {
                "model_type": "heuristic_fallback",
                "predicted_at": datetime.now().isoformat()
            }
        }
    
    def _fallback_performance_prediction(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback performance prediction when ML is not available"""
        current_perf = current_metrics.get('current_performance', 0.5)
        error_rate = current_metrics.get('error_rate', 0.0)
        
        # Simple prediction based on current state
        predicted_perf = current_perf * (1 - error_rate)
        
        return {
            "predicted_final_performance": predicted_perf,
            "current_performance": current_perf,
            "performance_trend": "stable",
            "confidence_interval": {"confidence": 0.5, "margin_of_error": 0.2},
            "improvement_potential": max(0, 1 - predicted_perf),
            "suggestions": ["Install scikit-learn for advanced performance predictions"],
            "prediction_metadata": {
                "model_type": "heuristic_fallback",
                "predicted_at": datetime.now().isoformat()
            }
        }
    
    def _fallback_activity_forecast(self, user_id: str, forecast_days: int) -> Dict[str, Any]:
        """Fallback activity forecast when ML is not available"""
        return {
            "user_id": user_id,
            "forecast_period_days": forecast_days,
            "forecast": "fallback_mode",
            "message": "Install scikit-learn for advanced activity forecasting",
            "simple_forecast": {
                "expected_daily_sessions": 2,
                "confidence": "low"
            }
        }