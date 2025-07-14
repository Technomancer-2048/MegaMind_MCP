#!/usr/bin/env python3
"""
Phase 8: Advanced Predictive Analytics Platform
Deep learning-powered predictive analytics for intelligent knowledge management
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, deque
import threading
import statistics
from dataclasses import dataclass, asdict
import uuid
import pickle

# Advanced ML and deep learning imports
try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logging.warning("Advanced ML libraries not available - using fallback predictive analytics")

logger = logging.getLogger(__name__)

@dataclass
class PredictiveModel:
    """Predictive model metadata"""
    model_id: str
    model_name: str
    model_type: str  # 'session_success', 'user_behavior', 'system_load', 'knowledge_evolution'
    algorithm: str   # 'random_forest', 'neural_network', 'gradient_boost', 'linear'
    features: List[str]
    target_variable: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_data_size: int
    last_trained: datetime
    prediction_count: int
    status: str  # 'active', 'training', 'deprecated'
    metadata: Dict[str, Any]

@dataclass
class Prediction:
    """Individual prediction result"""
    prediction_id: str
    model_id: str
    input_features: Dict[str, Any]
    prediction: Any
    confidence: float
    prediction_type: str
    created_at: datetime
    actual_outcome: Optional[Any] = None
    accuracy: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class TrendAnalysis:
    """Trend analysis result"""
    trend_id: str
    metric_name: str
    trend_type: str  # 'linear', 'exponential', 'cyclical', 'volatile'
    direction: str   # 'increasing', 'decreasing', 'stable'
    strength: float  # 0.0 to 1.0
    confidence: float
    forecast_values: List[float]
    forecast_timestamps: List[datetime]
    analysis_period: timedelta
    created_at: datetime

class DeepLearningPredictor:
    """
    Deep learning predictor using neural networks for complex pattern recognition
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        
        # Model configurations
        self.model_configs = {
            'session_success': {
                'hidden_layers': (100, 50, 25),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'learning_rate': 'adaptive'
            },
            'user_behavior': {
                'hidden_layers': (150, 75, 25),
                'activation': 'tanh',
                'solver': 'adam',
                'max_iter': 1500,
                'learning_rate': 'adaptive'
            },
            'knowledge_quality': {
                'hidden_layers': (80, 40),
                'activation': 'relu',
                'solver': 'lbfgs',
                'max_iter': 800
            }
        }
        
        logger.info("✅ Deep Learning Predictor initialized")
    
    def train_model(self, model_type: str, training_data: List[Dict[str, Any]], 
                   target_column: str) -> Optional[str]:
        """Train a deep learning model"""
        if not ML_AVAILABLE or not training_data:
            return None
        
        try:
            # Prepare features and target
            features_df = self._prepare_features(training_data, model_type)
            target_values = [item[target_column] for item in training_data]
            
            # Determine if classification or regression
            is_classification = self._is_classification_problem(target_values)
            
            # Create and configure model
            config = self.model_configs.get(model_type, self.model_configs['session_success'])
            
            if is_classification:
                model = MLPClassifier(**config, random_state=42)
            else:
                model = MLPRegressor(**config, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(features_df)
            
            # Encode target if classification
            if is_classification:
                encoder = LabelEncoder()
                y_encoded = encoder.fit_transform(target_values)
                self.encoders[model_type] = encoder
            else:
                y_encoded = np.array(target_values)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            
            if is_classification:
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            else:
                # For regression, use R² as accuracy
                accuracy = model.score(X_test, y_test)
                precision = accuracy  # Simplified for regression
                recall = accuracy
                f1 = accuracy
            
            # Store model and metadata
            model_id = str(uuid.uuid4())
            self.models[model_id] = model
            self.scalers[model_id] = scaler
            self.feature_columns[model_id] = list(features_df.columns)
            
            model_metadata = PredictiveModel(
                model_id=model_id,
                model_name=f"Deep Learning {model_type.title()}",
                model_type=model_type,
                algorithm='neural_network',
                features=list(features_df.columns),
                target_variable=target_column,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                training_data_size=len(training_data),
                last_trained=datetime.now(),
                prediction_count=0,
                status='active',
                metadata={
                    'is_classification': is_classification,
                    'hidden_layers': config['hidden_layers'],
                    'n_iter': model.n_iter_ if hasattr(model, 'n_iter_') else 0,
                    'loss': model.loss_ if hasattr(model, 'loss_') else 0
                }
            )
            
            logger.info(f"✅ Trained {model_type} model - Accuracy: {accuracy:.3f}")
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to train deep learning model: {e}")
            return None
    
    def _prepare_features(self, data: List[Dict[str, Any]], model_type: str) -> np.ndarray:
        """Prepare feature matrix from training data"""
        if not data:
            return np.array([])
        
        # Extract numerical features
        feature_names = set()
        for item in data:
            for key, value in item.items():
                if isinstance(value, (int, float)) and key != 'target':
                    feature_names.add(key)
        
        feature_names = sorted(list(feature_names))
        
        # Create feature matrix
        features = []
        for item in data:
            row = []
            for feature in feature_names:
                value = item.get(feature, 0)
                row.append(float(value) if value is not None else 0.0)
            features.append(row)
        
        return np.array(features)
    
    def _is_classification_problem(self, target_values: List[Any]) -> bool:
        """Determine if this is a classification or regression problem"""
        # Check if target values are categorical or boolean
        unique_values = set(target_values)
        
        # If all values are boolean or small set of discrete values
        if all(isinstance(v, bool) for v in target_values):
            return True
        
        if len(unique_values) <= 10 and all(isinstance(v, (int, str)) for v in target_values):
            return True
        
        return False
    
    def predict(self, model_id: str, features: Dict[str, Any]) -> Optional[Prediction]:
        """Make prediction using trained model"""
        if model_id not in self.models:
            return None
        
        try:
            model = self.models[model_id]
            scaler = self.scalers[model_id]
            feature_cols = self.feature_columns[model_id]
            
            # Prepare feature vector
            feature_vector = []
            for col in feature_cols:
                value = features.get(col, 0)
                feature_vector.append(float(value) if value is not None else 0.0)
            
            # Scale features
            X = scaler.transform([feature_vector])
            
            # Make prediction
            prediction = model.predict(X)[0]
            
            # Get confidence (prediction probability for classification)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[0]
                confidence = float(np.max(probabilities))
                
                # Decode prediction if classification
                if model_id in self.encoders:
                    prediction = self.encoders[model_id].inverse_transform([int(prediction)])[0]
            else:
                # For regression, use simplified confidence based on model score
                confidence = 0.8  # Default confidence for regression
            
            return Prediction(
                prediction_id=str(uuid.uuid4()),
                model_id=model_id,
                input_features=features,
                prediction=prediction,
                confidence=confidence,
                prediction_type='deep_learning',
                created_at=datetime.now(),
                metadata={'feature_vector': feature_vector}
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information"""
        if model_id not in self.models:
            return None
        
        model = self.models[model_id]
        
        info = {
            'model_id': model_id,
            'algorithm': 'neural_network',
            'features': self.feature_columns.get(model_id, []),
            'layers': model.hidden_layer_sizes if hasattr(model, 'hidden_layer_sizes') else [],
            'activation': model.activation if hasattr(model, 'activation') else 'unknown',
            'solver': model.solver if hasattr(model, 'solver') else 'unknown',
            'n_iter': model.n_iter_ if hasattr(model, 'n_iter_') else 0,
            'loss': model.loss_ if hasattr(model, 'loss_') else 0
        }
        
        return info

class TrendAnalysisEngine:
    """
    Advanced trend analysis and forecasting engine
    """
    
    def __init__(self):
        self.trend_models = {}
        self.historical_data = defaultdict(lambda: deque(maxlen=10000))
        self.trend_cache = {}
        
        # Trend analysis parameters
        self.min_data_points = 10
        self.forecast_periods = 30
        self.confidence_threshold = 0.7
        
        logger.info("✅ Trend Analysis Engine initialized")
    
    def analyze_trend(self, metric_name: str, data_points: List[Tuple[datetime, float]],
                     analysis_period: Optional[timedelta] = None) -> Optional[TrendAnalysis]:
        """Analyze trend in time series data"""
        if len(data_points) < self.min_data_points:
            return None
        
        try:
            # Sort data by timestamp
            sorted_data = sorted(data_points, key=lambda x: x[0])
            
            # Extract values and timestamps
            timestamps = [point[0] for point in sorted_data]
            values = [point[1] for point in sorted_data]
            
            # Convert timestamps to numerical format for analysis
            base_time = timestamps[0]
            time_deltas = [(ts - base_time).total_seconds() for ts in timestamps]
            
            # Perform trend analysis
            trend_type, direction, strength, confidence = self._analyze_trend_pattern(time_deltas, values)
            
            # Generate forecast
            forecast_times, forecast_values = self._generate_forecast(
                time_deltas, values, self.forecast_periods
            )
            
            # Convert forecast times back to datetime
            forecast_timestamps = [
                base_time + timedelta(seconds=t) for t in forecast_times
            ]
            
            return TrendAnalysis(
                trend_id=str(uuid.uuid4()),
                metric_name=metric_name,
                trend_type=trend_type,
                direction=direction,
                strength=strength,
                confidence=confidence,
                forecast_values=forecast_values,
                forecast_timestamps=forecast_timestamps,
                analysis_period=analysis_period or (timestamps[-1] - timestamps[0]),
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return None
    
    def _analyze_trend_pattern(self, time_values: List[float], 
                              data_values: List[float]) -> Tuple[str, str, float, float]:
        """Analyze the pattern in time series data"""
        if not ML_AVAILABLE:
            return self._simple_trend_analysis(data_values)
        
        try:
            # Convert to numpy arrays
            X = np.array(time_values).reshape(-1, 1)
            y = np.array(data_values)
            
            # Try different trend models
            models = {
                'linear': LinearRegression(),
                'polynomial': None,  # Will be implemented if needed
            }
            
            best_model = None
            best_score = -float('inf')
            best_type = 'linear'
            
            for model_type, model in models.items():
                if model is None:
                    continue
                
                try:
                    # Fit model
                    model.fit(X, y)
                    score = model.score(X, y)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_type = model_type
                
                except Exception:
                    continue
            
            if best_model is None:
                return self._simple_trend_analysis(data_values)
            
            # Determine direction and strength
            if best_type == 'linear':
                slope = best_model.coef_[0]
                direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                strength = min(1.0, abs(slope) / (max(data_values) - min(data_values)) * len(data_values))
            else:
                direction = 'stable'
                strength = 0.5
            
            confidence = max(0.0, min(1.0, best_score))
            
            return best_type, direction, strength, confidence
            
        except Exception as e:
            logger.warning(f"ML trend analysis failed: {e}, using simple analysis")
            return self._simple_trend_analysis(data_values)
    
    def _simple_trend_analysis(self, values: List[float]) -> Tuple[str, str, float, float]:
        """Simple trend analysis without ML"""
        if len(values) < 3:
            return 'linear', 'stable', 0.0, 0.5
        
        # Calculate simple linear trend
        n = len(values)
        x = list(range(n))
        
        # Calculate slope using least squares
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(xi * yi for xi, yi in zip(x, values))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = 'stable'
        elif slope > 0:
            direction = 'increasing'
        else:
            direction = 'decreasing'
        
        # Calculate strength and confidence
        value_range = max(values) - min(values)
        strength = min(1.0, abs(slope) * n / value_range) if value_range > 0 else 0.0
        
        # Simple confidence based on consistency
        half_point = n // 2
        first_half_avg = statistics.mean(values[:half_point])
        second_half_avg = statistics.mean(values[half_point:])
        
        if direction == 'increasing':
            confidence = 0.8 if second_half_avg > first_half_avg else 0.4
        elif direction == 'decreasing':
            confidence = 0.8 if second_half_avg < first_half_avg else 0.4
        else:
            confidence = 0.6
        
        return 'linear', direction, strength, confidence
    
    def _generate_forecast(self, time_values: List[float], data_values: List[float],
                          periods: int) -> Tuple[List[float], List[float]]:
        """Generate forecast values"""
        if not ML_AVAILABLE:
            return self._simple_forecast(time_values, data_values, periods)
        
        try:
            # Use linear regression for forecasting
            X = np.array(time_values).reshape(-1, 1)
            y = np.array(data_values)
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate future time points
            last_time = time_values[-1]
            time_step = (time_values[-1] - time_values[0]) / len(time_values)
            
            future_times = []
            for i in range(1, periods + 1):
                future_times.append(last_time + time_step * i)
            
            # Predict future values
            future_X = np.array(future_times).reshape(-1, 1)
            future_values = model.predict(future_X).tolist()
            
            return future_times, future_values
            
        except Exception as e:
            logger.warning(f"ML forecast failed: {e}, using simple forecast")
            return self._simple_forecast(time_values, data_values, periods)
    
    def _simple_forecast(self, time_values: List[float], data_values: List[float],
                        periods: int) -> Tuple[List[float], List[float]]:
        """Simple linear extrapolation forecast"""
        if len(data_values) < 2:
            return [], []
        
        # Calculate simple trend
        recent_values = data_values[-5:]  # Use last 5 points
        trend = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        # Generate forecast
        last_time = time_values[-1]
        time_step = (time_values[-1] - time_values[0]) / len(time_values)
        last_value = data_values[-1]
        
        future_times = []
        future_values = []
        
        for i in range(1, periods + 1):
            future_times.append(last_time + time_step * i)
            future_values.append(last_value + trend * i)
        
        return future_times, future_values
    
    def detect_anomalies(self, metric_name: str, recent_values: List[float],
                        threshold_std: float = 2.0) -> List[int]:
        """Detect anomalies in recent data"""
        if len(recent_values) < 10:
            return []
        
        try:
            # Calculate moving statistics
            mean_val = statistics.mean(recent_values)
            std_val = statistics.stdev(recent_values)
            
            # Find anomalies
            anomalies = []
            for i, value in enumerate(recent_values):
                if abs(value - mean_val) > threshold_std * std_val:
                    anomalies.append(i)
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []

class PredictiveAnalyticsPlatform:
    """
    Main predictive analytics platform coordinator
    """
    
    def __init__(self, db_manager, session_manager=None, ml_engine=None):
        self.db_manager = db_manager
        self.session_manager = session_manager
        self.ml_engine = ml_engine
        
        # Initialize components
        self.deep_learning_predictor = DeepLearningPredictor()
        self.trend_analyzer = TrendAnalysisEngine()
        
        # Model registry
        self.active_models = {}
        self.model_performance = defaultdict(list)
        self.prediction_history = deque(maxlen=10000)
        
        # Analytics data storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=5000))
        self.user_behavior_data = defaultdict(lambda: deque(maxlen=1000))
        self.system_metrics = deque(maxlen=2000)
        
        # Background processing
        self.is_running = False
        self.analytics_thread = None
        self.prediction_queue = asyncio.Queue(maxsize=1000)
        
        # Configuration
        self.model_retrain_interval = timedelta(days=7)
        self.prediction_confidence_threshold = 0.6
        
        logger.info("✅ Predictive Analytics Platform initialized")
    
    def start_analytics(self):
        """Start predictive analytics platform"""
        if self.is_running:
            logger.warning("⚠️ Predictive analytics already running")
            return
        
        self.is_running = True
        self.analytics_thread = threading.Thread(target=self._analytics_loop, daemon=True)
        self.analytics_thread.start()
        
        logger.info("🚀 Predictive Analytics Platform started")
    
    def stop_analytics(self):
        """Stop predictive analytics platform"""
        self.is_running = False
        
        if self.analytics_thread:
            self.analytics_thread.join(timeout=5.0)
        
        logger.info("⏹️ Predictive Analytics Platform stopped")
    
    def _analytics_loop(self):
        """Background analytics processing loop"""
        last_model_check = datetime.now()
        
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check if models need retraining
                if (current_time - last_model_check) >= self.model_retrain_interval:
                    asyncio.run(self._check_model_retraining())
                    last_model_check = current_time
                
                # Process prediction queue
                asyncio.run(self._process_prediction_queue())
                
                # Update trend analyses
                self._update_trend_analyses()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                time.sleep(300)  # Sleep 5 minutes on error
    
    async def _check_model_retraining(self):
        """Check if models need retraining"""
        for model_id, model_info in self.active_models.items():
            try:
                # Check if enough new data is available
                last_trained = model_info.last_trained
                if (datetime.now() - last_trained) >= self.model_retrain_interval:
                    
                    # Get new training data
                    new_data = await self._get_training_data(model_info.model_type)
                    
                    if len(new_data) >= 100:  # Minimum data for retraining
                        logger.info(f"🔄 Retraining model: {model_info.model_name}")
                        await self._retrain_model(model_id, new_data)
                
            except Exception as e:
                logger.error(f"Error checking model retraining: {e}")
    
    async def _process_prediction_queue(self):
        """Process queued prediction requests"""
        processed = 0
        
        while not self.prediction_queue.empty() and processed < 50:
            try:
                prediction_request = await asyncio.wait_for(
                    self.prediction_queue.get(), timeout=0.1
                )
                
                # Process prediction
                result = await self._make_prediction(prediction_request)
                if result:
                    self.prediction_history.append(result)
                
                processed += 1
                
            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing prediction: {e}")
    
    def _update_trend_analyses(self):
        """Update trend analyses for key metrics"""
        try:
            # Analyze trends for each metric
            for metric_name, data_points in self.metrics_history.items():
                if len(data_points) >= self.trend_analyzer.min_data_points:
                    
                    # Convert to timestamp-value pairs
                    time_series_data = [
                        (point.get('timestamp', datetime.now()), point.get('value', 0.0))
                        for point in data_points
                        if isinstance(point, dict) and 'value' in point
                    ]
                    
                    if len(time_series_data) >= self.trend_analyzer.min_data_points:
                        trend_analysis = self.trend_analyzer.analyze_trend(
                            metric_name, time_series_data
                        )
                        
                        if trend_analysis and trend_analysis.confidence > self.prediction_confidence_threshold:
                            logger.info(f"📈 Trend detected in {metric_name}: "
                                       f"{trend_analysis.direction} ({trend_analysis.confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Error updating trend analyses: {e}")
    
    async def train_session_success_model(self, training_data: List[Dict[str, Any]]) -> Optional[str]:
        """Train session success prediction model"""
        logger.info("🧠 Training session success prediction model")
        
        # Prepare training data
        processed_data = []
        for session in training_data:
            features = {
                'query_length': len(session.get('query', '').split()),
                'user_experience_level': session.get('user_experience', 0.5),
                'session_duration': session.get('duration_minutes', 0),
                'chunks_accessed': session.get('chunks_accessed', 0),
                'model_type_score': self._get_model_capability_score(session.get('model_type', 'sonnet')),
                'time_of_day': datetime.fromisoformat(session.get('timestamp', datetime.now().isoformat())).hour,
                'complexity_score': session.get('complexity_score', 0.5),
                'target': 1 if session.get('success', False) else 0
            }
            processed_data.append(features)
        
        # Train model
        model_id = self.deep_learning_predictor.train_model(
            'session_success', processed_data, 'target'
        )
        
        if model_id:
            # Register model
            model_info = PredictiveModel(
                model_id=model_id,
                model_name="Session Success Predictor",
                model_type='session_success',
                algorithm='neural_network',
                features=['query_length', 'user_experience_level', 'session_duration', 
                         'chunks_accessed', 'model_type_score', 'time_of_day', 'complexity_score'],
                target_variable='success',
                accuracy=0.0,  # Will be updated after first predictions
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                training_data_size=len(training_data),
                last_trained=datetime.now(),
                prediction_count=0,
                status='active',
                metadata={'model_type': 'session_success'}
            )
            
            self.active_models[model_id] = model_info
            logger.info(f"✅ Session success model trained: {model_id}")
        
        return model_id
    
    async def predict_session_success(self, session_context: Dict[str, Any]) -> Optional[Prediction]:
        """Predict session success probability"""
        # Find session success model
        session_model = None
        for model_id, model_info in self.active_models.items():
            if model_info.model_type == 'session_success':
                session_model = model_id
                break
        
        if not session_model:
            return None
        
        # Prepare features
        features = {
            'query_length': len(session_context.get('query', '').split()),
            'user_experience_level': session_context.get('user_experience', 0.5),
            'session_duration': session_context.get('duration_minutes', 0),
            'chunks_accessed': session_context.get('chunks_accessed', 0),
            'model_type_score': self._get_model_capability_score(session_context.get('model_type', 'sonnet')),
            'time_of_day': datetime.now().hour,
            'complexity_score': session_context.get('complexity_score', 0.5)
        }
        
        # Make prediction
        prediction = self.deep_learning_predictor.predict(session_model, features)
        
        if prediction:
            # Update model usage count
            self.active_models[session_model].prediction_count += 1
            
            # Queue for background processing
            try:
                await self.prediction_queue.put({
                    'type': 'session_success',
                    'prediction': prediction,
                    'context': session_context
                })
            except asyncio.QueueFull:
                logger.warning("Prediction queue full")
        
        return prediction
    
    async def predict_user_behavior(self, user_context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict user behavior patterns"""
        predictions = {
            'session_duration_minutes': 0,
            'likely_queries': [],
            'optimal_model_type': 'sonnet',
            'success_probability': 0.5,
            'confidence': 0.0
        }
        
        try:
            user_id = user_context.get('user_id')
            if user_id and user_id in self.user_behavior_data:
                historical_data = list(self.user_behavior_data[user_id])
                
                if len(historical_data) >= 5:
                    # Calculate averages from historical data
                    avg_duration = statistics.mean(
                        session.get('duration_minutes', 0) for session in historical_data
                    )
                    
                    # Find most successful model type
                    model_success = defaultdict(list)
                    for session in historical_data:
                        model_type = session.get('model_type', 'sonnet')
                        success = session.get('success', False)
                        model_success[model_type].append(success)
                    
                    best_model = 'sonnet'
                    best_success_rate = 0
                    for model, successes in model_success.items():
                        success_rate = sum(successes) / len(successes)
                        if success_rate > best_success_rate:
                            best_success_rate = success_rate
                            best_model = model
                    
                    predictions.update({
                        'session_duration_minutes': avg_duration,
                        'optimal_model_type': best_model,
                        'success_probability': best_success_rate,
                        'confidence': min(1.0, len(historical_data) / 20)
                    })
                    
                    # Extract common query patterns
                    query_words = []
                    for session in historical_data:
                        query = session.get('query', '')
                        query_words.extend(query.lower().split())
                    
                    if query_words:
                        word_counts = defaultdict(int)
                        for word in query_words:
                            if len(word) > 3:  # Skip short words
                                word_counts[word] += 1
                        
                        # Get top words as likely query themes
                        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                        predictions['likely_queries'] = [word for word, count in top_words]
        
        except Exception as e:
            logger.error(f"User behavior prediction failed: {e}")
        
        return predictions
    
    async def predict_system_load(self, time_horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict system load and resource requirements"""
        try:
            # Analyze historical system metrics
            if len(self.system_metrics) < 10:
                return {'message': 'Insufficient data for load prediction'}
            
            # Extract relevant metrics
            recent_metrics = list(self.system_metrics)[-100:]  # Last 100 data points
            
            cpu_usage = [m.get('cpu_usage', 0) for m in recent_metrics]
            memory_usage = [m.get('memory_usage', 0) for m in recent_metrics]
            request_count = [m.get('request_count', 0) for m in recent_metrics]
            response_time = [m.get('response_time_ms', 0) for m in recent_metrics]
            
            # Analyze trends
            time_points = [(datetime.now() - timedelta(minutes=i), value) 
                          for i, value in enumerate(reversed(cpu_usage))]
            
            cpu_trend = self.trend_analyzer.analyze_trend('cpu_usage', time_points)
            
            # Generate predictions
            predictions = {
                'time_horizon_hours': time_horizon_hours,
                'predicted_peak_cpu': max(cpu_usage) * 1.1 if cpu_usage else 0,
                'predicted_peak_memory': max(memory_usage) * 1.1 if memory_usage else 0,
                'predicted_request_volume': int(statistics.mean(request_count) * 1.2) if request_count else 0,
                'predicted_response_time': statistics.mean(response_time) * 1.1 if response_time else 0,
                'confidence': 0.7,
                'recommendations': []
            }
            
            # Add recommendations based on trends
            if cpu_trend and cpu_trend.direction == 'increasing':
                predictions['recommendations'].append("Consider CPU scaling due to increasing trend")
            
            if predictions['predicted_peak_memory'] > 80:
                predictions['recommendations'].append("Memory usage may approach limits")
            
            return predictions
            
        except Exception as e:
            logger.error(f"System load prediction failed: {e}")
            return {'error': str(e)}
    
    def record_session_data(self, session_data: Dict[str, Any]):
        """Record session data for analysis"""
        try:
            user_id = session_data.get('user_id')
            if user_id:
                self.user_behavior_data[user_id].append(session_data)
            
            # Record session metrics
            session_metrics = {
                'timestamp': datetime.now(),
                'value': 1 if session_data.get('success', False) else 0,
                'metadata': session_data
            }
            
            self.metrics_history['session_success'].append(session_metrics)
            
        except Exception as e:
            logger.error(f"Failed to record session data: {e}")
    
    def record_system_metrics(self, metrics: Dict[str, Any]):
        """Record system performance metrics"""
        try:
            metrics['timestamp'] = datetime.now()
            self.system_metrics.append(metrics)
            
            # Record individual metrics for trend analysis
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and metric_name != 'timestamp':
                    metric_data = {
                        'timestamp': metrics['timestamp'],
                        'value': value
                    }
                    self.metrics_history[metric_name].append(metric_data)
            
        except Exception as e:
            logger.error(f"Failed to record system metrics: {e}")
    
    def _get_model_capability_score(self, model_type: str) -> float:
        """Get normalized model capability score"""
        capabilities = {
            'haiku': 0.3,
            'sonnet': 0.7,
            'opus': 1.0
        }
        return capabilities.get(model_type.lower(), 0.5)
    
    async def _get_training_data(self, model_type: str) -> List[Dict[str, Any]]:
        """Get training data for model retraining"""
        # This would integrate with the actual database
        # For now, return empty list as placeholder
        return []
    
    async def _retrain_model(self, model_id: str, training_data: List[Dict[str, Any]]):
        """Retrain existing model with new data"""
        try:
            model_info = self.active_models[model_id]
            
            # Retrain the model
            new_model_id = self.deep_learning_predictor.train_model(
                model_info.model_type, training_data, model_info.target_variable
            )
            
            if new_model_id:
                # Update model info
                model_info.model_id = new_model_id
                model_info.last_trained = datetime.now()
                model_info.training_data_size = len(training_data)
                
                # Remove old model from predictor
                if model_id in self.deep_learning_predictor.models:
                    del self.deep_learning_predictor.models[model_id]
                
                # Update registry
                self.active_models[new_model_id] = model_info
                del self.active_models[model_id]
                
                logger.info(f"✅ Model retrained: {model_info.model_name}")
        
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
    
    async def _make_prediction(self, prediction_request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process prediction request"""
        try:
            prediction_type = prediction_request.get('type')
            prediction = prediction_request.get('prediction')
            context = prediction_request.get('context', {})
            
            # Process based on type
            result = {
                'prediction_id': prediction.prediction_id,
                'type': prediction_type,
                'prediction': prediction.prediction,
                'confidence': prediction.confidence,
                'timestamp': prediction.created_at.isoformat(),
                'processed': True
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing prediction request: {e}")
            return None
    
    def get_analytics_status(self) -> Dict[str, Any]:
        """Get predictive analytics status"""
        return {
            'analytics_active': self.is_running,
            'active_models': len(self.active_models),
            'prediction_history_size': len(self.prediction_history),
            'metrics_tracked': len(self.metrics_history),
            'prediction_queue_size': self.prediction_queue.qsize() if hasattr(self.prediction_queue, 'qsize') else 0,
            'ml_available': ML_AVAILABLE,
            'model_types': list(set(model.model_type for model in self.active_models.values()))
        }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance summary"""
        performance = {}
        
        for model_id, model_info in self.active_models.items():
            performance[model_id] = {
                'model_name': model_info.model_name,
                'model_type': model_info.model_type,
                'accuracy': model_info.accuracy,
                'precision': model_info.precision,
                'recall': model_info.recall,
                'f1_score': model_info.f1_score,
                'prediction_count': model_info.prediction_count,
                'last_trained': model_info.last_trained.isoformat(),
                'status': model_info.status
            }
        
        return performance