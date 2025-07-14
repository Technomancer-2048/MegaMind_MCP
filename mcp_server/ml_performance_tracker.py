#!/usr/bin/env python3
"""
Phase 7: ML Model Performance Tracker
Live tracking and alerting for ML model performance with real-time monitoring
"""

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
import threading
import statistics
from dataclasses import dataclass, asdict
import uuid

# ML imports for performance tracking
try:
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("ML libraries not available - using fallback performance tracking")

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Individual model prediction record"""
    prediction_id: str
    model_name: str
    model_version: str
    input_features: Dict[str, Any]
    prediction: Any
    confidence: Optional[float]
    timestamp: datetime
    actual_outcome: Optional[Any] = None
    prediction_time_ms: Optional[float] = None

@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics snapshot"""
    model_name: str
    model_version: str
    timestamp: datetime
    prediction_count: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    avg_prediction_time: Optional[float] = None
    confidence_distribution: Optional[Dict[str, float]] = None

@dataclass
class PerformanceAlert:
    """Model performance alert"""
    alert_id: str
    model_name: str
    alert_type: str  # 'degradation', 'drift', 'latency', 'accuracy_drop'
    severity: str    # 'low', 'medium', 'high', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    recommended_actions: List[str]

class MLModelPerformanceTracker:
    """
    Real-time ML model performance tracking and alerting system
    Monitors prediction accuracy, latency, and model drift
    """
    
    def __init__(self, ml_engine=None):
        self.ml_engine = ml_engine
        
        # Prediction tracking
        self.predictions = defaultdict(lambda: deque(maxlen=10000))  # model_name -> predictions
        self.prediction_history = deque(maxlen=50000)  # Global prediction history
        
        # Performance metrics tracking
        self.performance_metrics = defaultdict(lambda: deque(maxlen=1000))  # model_name -> metrics
        self.current_metrics = {}  # model_name -> latest metrics
        
        # Alert system
        self.alerts = deque(maxlen=1000)
        self.alert_thresholds = {
            'accuracy_threshold': 0.8,
            'latency_threshold_ms': 1000,
            'confidence_threshold': 0.6,
            'drift_threshold': 0.1,
            'error_rate_threshold': 0.1
        }
        
        # Model metadata
        self.model_registry = {}  # model_name -> metadata
        self.model_versions = defaultdict(str)  # model_name -> current version
        
        # Performance calculation settings
        self.metrics_window_minutes = 30
        self.metrics_calculation_interval = 60  # seconds
        
        # Background processing
        self.is_running = False
        self.processing_thread = None
        self.lock = threading.RLock()
        
        logger.info("✅ ML Model Performance Tracker initialized")
    
    def start_tracking(self):
        """Start the performance tracking system"""
        if self.is_running:
            logger.warning("⚠️ Performance tracking already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._performance_calculation_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("🚀 ML Model Performance Tracking started")
    
    def stop_tracking(self):
        """Stop the performance tracking system"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("⏹️ ML Model Performance Tracking stopped")
    
    def register_model(self, model_name: str, model_type: str, version: str = "1.0", 
                      metadata: Dict[str, Any] = None):
        """Register a model for performance tracking"""
        self.model_registry[model_name] = {
            'model_type': model_type,  # 'classifier', 'regressor', 'clustering'
            'version': version,
            'registered_at': datetime.now(),
            'metadata': metadata or {}
        }
        self.model_versions[model_name] = version
        
        logger.info(f"📊 Registered model for tracking: {model_name} v{version}")
    
    def record_prediction(self, model_name: str, input_features: Dict[str, Any], 
                         prediction: Any, confidence: Optional[float] = None,
                         prediction_time_ms: Optional[float] = None) -> str:
        """Record a model prediction for performance tracking"""
        with self.lock:
            prediction_record = ModelPrediction(
                prediction_id=str(uuid.uuid4()),
                model_name=model_name,
                model_version=self.model_versions.get(model_name, "unknown"),
                input_features=input_features,
                prediction=prediction,
                confidence=confidence,
                timestamp=datetime.now(),
                prediction_time_ms=prediction_time_ms
            )
            
            # Store prediction
            self.predictions[model_name].append(prediction_record)
            self.prediction_history.append(prediction_record)
            
            # Check for immediate alerts (latency, confidence)
            self._check_immediate_alerts(prediction_record)
            
            return prediction_record.prediction_id
    
    def record_actual_outcome(self, prediction_id: str, actual_outcome: Any):
        """Record the actual outcome for a prediction to calculate accuracy"""
        with self.lock:
            # Find the prediction in history
            for prediction in self.prediction_history:
                if prediction.prediction_id == prediction_id:
                    prediction.actual_outcome = actual_outcome
                    
                    # Trigger performance recalculation for this model
                    self._calculate_model_performance(prediction.model_name, force=True)
                    break
    
    def _performance_calculation_loop(self):
        """Background loop for calculating performance metrics"""
        last_calculation = {}
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Calculate metrics for each model
                for model_name in self.model_registry.keys():
                    last_calc = last_calculation.get(model_name, datetime.min)
                    
                    if (current_time - last_calc).total_seconds() >= self.metrics_calculation_interval:
                        self._calculate_model_performance(model_name)
                        last_calculation[model_name] = current_time
                
                # Sleep for a short interval
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in performance calculation loop: {e}")
                time.sleep(30)
    
    def _calculate_model_performance(self, model_name: str, force: bool = False):
        """Calculate performance metrics for a specific model"""
        try:
            predictions = list(self.predictions[model_name])
            if not predictions:
                return
            
            # Filter to recent predictions within window
            window_start = datetime.now() - timedelta(minutes=self.metrics_window_minutes)
            recent_predictions = [p for p in predictions if p.timestamp >= window_start]
            
            if len(recent_predictions) < 5 and not force:
                return  # Not enough data
            
            # Calculate metrics based on model type
            model_info = self.model_registry.get(model_name, {})
            model_type = model_info.get('model_type', 'classifier')
            
            metrics = self._calculate_metrics_by_type(recent_predictions, model_type)
            
            # Create performance metrics record
            performance_metrics = ModelPerformanceMetrics(
                model_name=model_name,
                model_version=self.model_versions.get(model_name, "unknown"),
                timestamp=datetime.now(),
                prediction_count=len(recent_predictions),
                **metrics
            )
            
            # Store metrics
            self.performance_metrics[model_name].append(performance_metrics)
            self.current_metrics[model_name] = performance_metrics
            
            # Check for performance alerts
            self._check_performance_alerts(model_name, performance_metrics)
            
            logger.info(f"📊 Updated performance metrics for {model_name}: "
                       f"{len(recent_predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error calculating performance for {model_name}: {e}")
    
    def _calculate_metrics_by_type(self, predictions: List[ModelPrediction], 
                                  model_type: str) -> Dict[str, Any]:
        """Calculate metrics based on model type"""
        metrics = {}
        
        # Get predictions with actual outcomes
        labeled_predictions = [p for p in predictions if p.actual_outcome is not None]
        
        if not labeled_predictions:
            metrics['avg_prediction_time'] = self._calculate_avg_prediction_time(predictions)
            metrics['confidence_distribution'] = self._calculate_confidence_distribution(predictions)
            return metrics
        
        # Extract predictions and actuals
        y_pred = [p.prediction for p in labeled_predictions]
        y_true = [p.actual_outcome for p in labeled_predictions]
        
        if not ML_AVAILABLE:
            metrics.update(self._fallback_metrics_calculation(y_pred, y_true, model_type))
        else:
            if model_type == 'classifier':
                metrics.update(self._calculate_classification_metrics(y_pred, y_true))
            elif model_type == 'regressor':
                metrics.update(self._calculate_regression_metrics(y_pred, y_true))
        
        # Common metrics
        metrics['avg_prediction_time'] = self._calculate_avg_prediction_time(predictions)
        metrics['confidence_distribution'] = self._calculate_confidence_distribution(predictions)
        
        return metrics
    
    def _calculate_classification_metrics(self, y_pred: List, y_true: List) -> Dict[str, float]:
        """Calculate classification metrics"""
        try:
            return {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                'f1_score': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            }
        except Exception as e:
            logger.warning(f"Error calculating classification metrics: {e}")
            return self._fallback_classification_metrics(y_pred, y_true)
    
    def _calculate_regression_metrics(self, y_pred: List, y_true: List) -> Dict[str, float]:
        """Calculate regression metrics"""
        try:
            y_pred_array = np.array(y_pred, dtype=float)
            y_true_array = np.array(y_true, dtype=float)
            
            return {
                'mse': float(mean_squared_error(y_true_array, y_pred_array)),
                'mae': float(mean_absolute_error(y_true_array, y_pred_array)),
                'r2_score': float(r2_score(y_true_array, y_pred_array))
            }
        except Exception as e:
            logger.warning(f"Error calculating regression metrics: {e}")
            return self._fallback_regression_metrics(y_pred, y_true)
    
    def _fallback_metrics_calculation(self, y_pred: List, y_true: List, 
                                    model_type: str) -> Dict[str, float]:
        """Fallback metrics calculation without scikit-learn"""
        if model_type == 'classifier':
            return self._fallback_classification_metrics(y_pred, y_true)
        elif model_type == 'regressor':
            return self._fallback_regression_metrics(y_pred, y_true)
        return {}
    
    def _fallback_classification_metrics(self, y_pred: List, y_true: List) -> Dict[str, float]:
        """Fallback classification metrics"""
        correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
        total = len(y_pred)
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': accuracy,  # Simplified
            'recall': accuracy,     # Simplified
            'f1_score': accuracy    # Simplified
        }
    
    def _fallback_regression_metrics(self, y_pred: List, y_true: List) -> Dict[str, float]:
        """Fallback regression metrics"""
        try:
            errors = [abs(p - t) for p, t in zip(y_pred, y_true)]
            mae = statistics.mean(errors) if errors else 0.0
            mse = statistics.mean([e**2 for e in errors]) if errors else 0.0
            
            return {
                'mse': mse,
                'mae': mae,
                'r2_score': 0.0  # Difficult to calculate without numpy
            }
        except Exception:
            return {'mse': 0.0, 'mae': 0.0, 'r2_score': 0.0}
    
    def _calculate_avg_prediction_time(self, predictions: List[ModelPrediction]) -> Optional[float]:
        """Calculate average prediction time"""
        times = [p.prediction_time_ms for p in predictions if p.prediction_time_ms is not None]
        return statistics.mean(times) if times else None
    
    def _calculate_confidence_distribution(self, predictions: List[ModelPrediction]) -> Dict[str, float]:
        """Calculate confidence score distribution"""
        confidences = [p.confidence for p in predictions if p.confidence is not None]
        
        if not confidences:
            return {}
        
        # Calculate distribution buckets
        high_conf = sum(1 for c in confidences if c >= 0.8)
        medium_conf = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low_conf = sum(1 for c in confidences if c < 0.5)
        total = len(confidences)
        
        return {
            'high_confidence': high_conf / total,
            'medium_confidence': medium_conf / total,
            'low_confidence': low_conf / total,
            'avg_confidence': statistics.mean(confidences)
        }
    
    def _check_immediate_alerts(self, prediction: ModelPrediction):
        """Check for immediate alerts (latency, confidence)"""
        # Check latency alert
        if (prediction.prediction_time_ms and 
            prediction.prediction_time_ms > self.alert_thresholds['latency_threshold_ms']):
            self._create_alert(
                model_name=prediction.model_name,
                alert_type='latency',
                severity='medium',
                message=f"High prediction latency: {prediction.prediction_time_ms:.2f}ms",
                metric_name='prediction_time_ms',
                current_value=prediction.prediction_time_ms,
                threshold_value=self.alert_thresholds['latency_threshold_ms']
            )
        
        # Check confidence alert
        if (prediction.confidence and 
            prediction.confidence < self.alert_thresholds['confidence_threshold']):
            self._create_alert(
                model_name=prediction.model_name,
                alert_type='low_confidence',
                severity='low',
                message=f"Low prediction confidence: {prediction.confidence:.3f}",
                metric_name='confidence',
                current_value=prediction.confidence,
                threshold_value=self.alert_thresholds['confidence_threshold']
            )
    
    def _check_performance_alerts(self, model_name: str, metrics: ModelPerformanceMetrics):
        """Check for performance degradation alerts"""
        # Check accuracy alert
        if metrics.accuracy is not None and metrics.accuracy < self.alert_thresholds['accuracy_threshold']:
            self._create_alert(
                model_name=model_name,
                alert_type='accuracy_drop',
                severity='high',
                message=f"Model accuracy below threshold: {metrics.accuracy:.3f}",
                metric_name='accuracy',
                current_value=metrics.accuracy,
                threshold_value=self.alert_thresholds['accuracy_threshold']
            )
        
        # Check for performance drift
        self._check_performance_drift(model_name, metrics)
    
    def _check_performance_drift(self, model_name: str, current_metrics: ModelPerformanceMetrics):
        """Check for model performance drift"""
        historical_metrics = list(self.performance_metrics[model_name])
        
        if len(historical_metrics) < 5:
            return  # Not enough history
        
        # Compare with historical average
        historical_accuracy = [m.accuracy for m in historical_metrics[-5:] if m.accuracy is not None]
        
        if len(historical_accuracy) >= 3 and current_metrics.accuracy is not None:
            historical_avg = statistics.mean(historical_accuracy)
            drift = abs(current_metrics.accuracy - historical_avg)
            
            if drift > self.alert_thresholds['drift_threshold']:
                severity = 'critical' if drift > 0.2 else 'high'
                self._create_alert(
                    model_name=model_name,
                    alert_type='drift',
                    severity=severity,
                    message=f"Model performance drift detected: {drift:.3f}",
                    metric_name='accuracy_drift',
                    current_value=drift,
                    threshold_value=self.alert_thresholds['drift_threshold']
                )
    
    def _create_alert(self, model_name: str, alert_type: str, severity: str, 
                     message: str, metric_name: str, current_value: float, 
                     threshold_value: float):
        """Create and store a performance alert"""
        # Generate recommendations based on alert type
        recommendations = self._generate_alert_recommendations(alert_type, model_name)
        
        alert = PerformanceAlert(
            alert_id=str(uuid.uuid4()),
            model_name=model_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=threshold_value,
            timestamp=datetime.now(),
            recommended_actions=recommendations
        )
        
        self.alerts.append(alert)
        
        logger.warning(f"🚨 ML Performance Alert - {model_name}: {message}")
    
    def _generate_alert_recommendations(self, alert_type: str, model_name: str) -> List[str]:
        """Generate recommendations for performance alerts"""
        recommendations = []
        
        if alert_type == 'accuracy_drop':
            recommendations.extend([
                "Investigate recent data quality changes",
                "Consider model retraining with recent data",
                "Check for feature distribution changes",
                "Review model hyperparameters"
            ])
        elif alert_type == 'latency':
            recommendations.extend([
                "Optimize model inference pipeline",
                "Check system resource utilization",
                "Consider model compression techniques",
                "Review database query performance"
            ])
        elif alert_type == 'drift':
            recommendations.extend([
                "Retrain model with recent data",
                "Investigate data distribution changes",
                "Update feature engineering pipeline",
                "Consider online learning approaches"
            ])
        elif alert_type == 'low_confidence':
            recommendations.extend([
                "Review input data quality",
                "Check for edge cases in predictions",
                "Consider ensemble methods",
                "Validate model calibration"
            ])
        
        return recommendations
    
    # ================================================================
    # PUBLIC API METHODS
    # ================================================================
    
    def get_model_performance(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get current performance metrics for a model"""
        current_metrics = self.current_metrics.get(model_name)
        if not current_metrics:
            return None
        
        return asdict(current_metrics)
    
    def get_all_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        return {
            model_name: asdict(metrics) 
            for model_name, metrics in self.current_metrics.items()
        }
    
    def get_performance_history(self, model_name: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance history for a model"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        history = [
            asdict(metrics) for metrics in self.performance_metrics[model_name]
            if metrics.timestamp >= cutoff_time
        ]
        
        return sorted(history, key=lambda x: x['timestamp'])
    
    def get_recent_alerts(self, model_name: Optional[str] = None, 
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent performance alerts"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        alerts = [
            asdict(alert) for alert in self.alerts
            if alert.timestamp >= cutoff_time and
            (model_name is None or alert.model_name == model_name)
        ]
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_tracking_status(self) -> Dict[str, Any]:
        """Get current tracking status"""
        return {
            'tracking_active': self.is_running,
            'registered_models': len(self.model_registry),
            'total_predictions': len(self.prediction_history),
            'active_alerts': len([a for a in self.alerts 
                                if a.timestamp > datetime.now() - timedelta(hours=1)]),
            'metrics_window_minutes': self.metrics_window_minutes,
            'thresholds': self.alert_thresholds.copy()
        }
    
    def update_alert_thresholds(self, thresholds: Dict[str, float]):
        """Update alert thresholds"""
        self.alert_thresholds.update(thresholds)
        logger.info(f"📊 Updated alert thresholds: {thresholds}")
    
    def clear_alerts(self, model_name: Optional[str] = None):
        """Clear alerts for a specific model or all models"""
        if model_name:
            self.alerts = deque(
                [alert for alert in self.alerts if alert.model_name != model_name],
                maxlen=1000
            )
        else:
            self.alerts.clear()
        
        logger.info(f"🧹 Cleared alerts for {model_name or 'all models'}")