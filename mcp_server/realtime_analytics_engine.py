#!/usr/bin/env python3
"""
Phase 7: Real-time Analytics Engine
Real-time monitoring and analytics with streaming data processing
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
import queue
import statistics

# WebSocket and async imports for real-time communication
try:
    import websockets
    import aiohttp
    from aiohttp import web
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("WebSocket libraries not available - using fallback real-time analytics")

# ML imports for real-time processing
try:
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RealTimeEvent:
    """Real-time event data structure"""
    event_id: str
    event_type: str
    session_id: str
    user_id: str
    timestamp: datetime
    data: Dict[str, Any]
    ml_features: Optional[Dict[str, float]] = None
    anomaly_score: Optional[float] = None

@dataclass
class RealTimeAlert:
    """Real-time alert data structure"""
    alert_id: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    source_event: RealTimeEvent
    threshold_violated: Optional[Dict[str, Any]] = None
    recommended_actions: List[str] = None

class RealTimeMetricsCollector:
    """
    Collects and aggregates real-time metrics with sliding windows
    """
    
    def __init__(self, window_size_minutes: int = 5, max_events: int = 10000):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.max_events = max_events
        self.events = deque(maxlen=max_events)
        self.metrics_cache = {}
        self.lock = threading.RLock()
        
        # Performance metrics tracking
        self.session_metrics = defaultdict(lambda: {
            'start_time': None,
            'last_activity': None,
            'operations_count': 0,
            'errors_count': 0,
            'performance_score': 0.0,
            'ml_predictions': []
        })
        
        # System-wide metrics
        self.system_metrics = {
            'active_sessions': set(),
            'total_operations': 0,
            'error_rate': 0.0,
            'average_response_time': 0.0,
            'ml_model_accuracy': 0.0,
            'anomalies_detected': 0
        }
    
    def add_event(self, event: RealTimeEvent):
        """Add new event to the metrics collector"""
        with self.lock:
            self.events.append(event)
            self._update_session_metrics(event)
            self._update_system_metrics(event)
            self._cleanup_old_events()
    
    def _update_session_metrics(self, event: RealTimeEvent):
        """Update session-specific metrics"""
        session_id = event.session_id
        session_data = self.session_metrics[session_id]
        
        # Update session timing
        if session_data['start_time'] is None:
            session_data['start_time'] = event.timestamp
        session_data['last_activity'] = event.timestamp
        
        # Update operation counts
        if event.event_type == 'operation':
            session_data['operations_count'] += 1
        elif event.event_type == 'error':
            session_data['errors_count'] += 1
        
        # Update performance score if available
        if 'performance_score' in event.data:
            session_data['performance_score'] = event.data['performance_score']
        
        # Store ML predictions
        if event.ml_features:
            session_data['ml_predictions'].append({
                'timestamp': event.timestamp,
                'features': event.ml_features,
                'anomaly_score': event.anomaly_score
            })
    
    def _update_system_metrics(self, event: RealTimeEvent):
        """Update system-wide metrics"""
        self.system_metrics['active_sessions'].add(event.session_id)
        self.system_metrics['total_operations'] += 1
        
        # Update error rate
        recent_events = [e for e in self.events if e.timestamp > datetime.now() - self.window_size]
        if recent_events:
            error_events = [e for e in recent_events if e.event_type == 'error']
            self.system_metrics['error_rate'] = len(error_events) / len(recent_events)
        
        # Update response time if available
        if 'response_time' in event.data:
            response_times = [e.data.get('response_time', 0) for e in recent_events 
                            if 'response_time' in e.data]
            if response_times:
                self.system_metrics['average_response_time'] = statistics.mean(response_times)
        
        # Count anomalies
        if event.anomaly_score and event.anomaly_score > 0.7:
            self.system_metrics['anomalies_detected'] += 1
    
    def _cleanup_old_events(self):
        """Remove events outside the time window"""
        cutoff_time = datetime.now() - self.window_size
        while self.events and self.events[0].timestamp < cutoff_time:
            self.events.popleft()
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current aggregated metrics"""
        with self.lock:
            return {
                'timestamp': datetime.now().isoformat(),
                'window_size_minutes': self.window_size.total_seconds() / 60,
                'total_events': len(self.events),
                'system_metrics': {
                    'active_sessions_count': len(self.system_metrics['active_sessions']),
                    'total_operations': self.system_metrics['total_operations'],
                    'error_rate': self.system_metrics['error_rate'],
                    'average_response_time': self.system_metrics['average_response_time'],
                    'anomalies_detected': self.system_metrics['anomalies_detected']
                },
                'session_count': len(self.session_metrics),
                'recent_activity': len([e for e in self.events 
                                     if e.timestamp > datetime.now() - timedelta(minutes=1)])
            }

class RealTimeAnomalyDetector:
    """
    Real-time anomaly detection using streaming ML algorithms
    """
    
    def __init__(self, sensitivity: float = 0.1):
        self.sensitivity = sensitivity
        self.feature_history = deque(maxlen=1000)
        self.anomaly_threshold = 0.7
        self.scaler = StandardScaler() if ML_AVAILABLE else None
        self.detection_model = None
        self.is_trained = False
        
        # Anomaly tracking
        self.recent_anomalies = deque(maxlen=100)
        self.anomaly_patterns = defaultdict(int)
    
    def detect_anomaly(self, event: RealTimeEvent) -> Optional[RealTimeAlert]:
        """Detect anomalies in real-time events"""
        try:
            # Extract features from event
            features = self._extract_anomaly_features(event)
            if not features:
                return None
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(features)
            event.anomaly_score = anomaly_score
            
            # Check if anomaly threshold exceeded
            if anomaly_score > self.anomaly_threshold:
                return self._create_anomaly_alert(event, anomaly_score, features)
            
            # Update training data
            self._update_training_data(features, anomaly_score < self.anomaly_threshold)
            
            return None
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return None
    
    def _extract_anomaly_features(self, event: RealTimeEvent) -> Optional[List[float]]:
        """Extract numerical features for anomaly detection"""
        try:
            features = []
            
            # Temporal features
            features.append(event.timestamp.hour)  # Hour of day
            features.append(event.timestamp.weekday())  # Day of week
            
            # Event type encoding
            event_type_map = {'operation': 1, 'error': 3, 'warning': 2, 'info': 0}
            features.append(event_type_map.get(event.event_type, 0))
            
            # Data features
            if 'response_time' in event.data:
                features.append(float(event.data['response_time']))
            else:
                features.append(0.0)
            
            if 'performance_score' in event.data:
                features.append(float(event.data['performance_score']))
            else:
                features.append(0.5)  # Default performance
            
            if 'error_count' in event.data:
                features.append(float(event.data['error_count']))
            else:
                features.append(0.0)
            
            # Session context features
            features.append(hash(event.session_id) % 1000 / 1000.0)  # Session fingerprint
            features.append(hash(event.user_id) % 1000 / 1000.0)     # User fingerprint
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract anomaly features: {e}")
            return None
    
    def _calculate_anomaly_score(self, features: List[float]) -> float:
        """Calculate anomaly score for feature vector"""
        if not ML_AVAILABLE or len(self.feature_history) < 50:
            return self._fallback_anomaly_score(features)
        
        try:
            # Use DBSCAN for density-based anomaly detection
            if not self.is_trained and len(self.feature_history) >= 100:
                self._train_anomaly_detector()
            
            if self.is_trained and self.detection_model:
                # Scale features
                scaled_features = self.scaler.transform([features])
                
                # Predict cluster (-1 indicates anomaly)
                cluster = self.detection_model.fit_predict(scaled_features)[0]
                
                if cluster == -1:
                    return 0.9  # High anomaly score
                else:
                    return 0.1  # Low anomaly score
            
            return self._fallback_anomaly_score(features)
            
        except Exception as e:
            logger.warning(f"ML anomaly detection failed: {e}")
            return self._fallback_anomaly_score(features)
    
    def _fallback_anomaly_score(self, features: List[float]) -> float:
        """Fallback anomaly detection using statistical methods"""
        if len(self.feature_history) < 10:
            return 0.1  # Not enough data
        
        try:
            # Calculate z-scores for each feature
            anomaly_scores = []
            
            for i, feature_value in enumerate(features):
                historical_values = [f[i] for f in self.feature_history if len(f) > i]
                if len(historical_values) < 5:
                    continue
                
                mean_val = statistics.mean(historical_values)
                std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 1.0
                
                if std_val > 0:
                    z_score = abs((feature_value - mean_val) / std_val)
                    anomaly_scores.append(min(z_score / 3.0, 1.0))  # Normalize to 0-1
            
            return statistics.mean(anomaly_scores) if anomaly_scores else 0.1
            
        except Exception as e:
            logger.warning(f"Fallback anomaly detection failed: {e}")
            return 0.1
    
    def _train_anomaly_detector(self):
        """Train the anomaly detection model"""
        try:
            if not ML_AVAILABLE:
                return
            
            # Prepare training data
            training_data = list(self.feature_history)
            if len(training_data) < 100:
                return
            
            # Scale features
            self.scaler.fit(training_data)
            scaled_data = self.scaler.transform(training_data)
            
            # Train DBSCAN model
            self.detection_model = DBSCAN(eps=0.5, min_samples=5)
            self.detection_model.fit(scaled_data)
            
            self.is_trained = True
            logger.info("✅ Real-time anomaly detector trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train anomaly detector: {e}")
            self.is_trained = False
    
    def _update_training_data(self, features: List[float], is_normal: bool):
        """Update training data with new features"""
        if is_normal:  # Only train on normal data
            self.feature_history.append(features)
    
    def _create_anomaly_alert(self, event: RealTimeEvent, anomaly_score: float, 
                            features: List[float]) -> RealTimeAlert:
        """Create anomaly alert"""
        # Determine severity based on anomaly score
        if anomaly_score > 0.9:
            severity = 'critical'
        elif anomaly_score > 0.8:
            severity = 'high'
        elif anomaly_score > 0.7:
            severity = 'medium'
        else:
            severity = 'low'
        
        # Generate alert message
        message = f"Anomaly detected in {event.event_type} event (score: {anomaly_score:.3f})"
        
        # Add context if available
        if 'response_time' in event.data and event.data['response_time'] > 5000:
            message += f" - High response time: {event.data['response_time']}ms"
        
        if event.event_type == 'error':
            message += f" - Error event in session {event.session_id}"
        
        # Generate recommendations
        recommendations = self._generate_anomaly_recommendations(event, anomaly_score)
        
        alert = RealTimeAlert(
            alert_id=f"anomaly_{int(time.time())}_{event.event_id}",
            alert_type='anomaly_detection',
            severity=severity,
            message=message,
            source_event=event,
            threshold_violated={'anomaly_score': anomaly_score, 'threshold': self.anomaly_threshold},
            recommended_actions=recommendations
        )
        
        self.recent_anomalies.append(alert)
        self.anomaly_patterns[event.event_type] += 1
        
        return alert
    
    def _generate_anomaly_recommendations(self, event: RealTimeEvent, 
                                        anomaly_score: float) -> List[str]:
        """Generate recommendations for handling anomalies"""
        recommendations = []
        
        if anomaly_score > 0.9:
            recommendations.append("Investigate immediately - critical anomaly detected")
        
        if event.event_type == 'error':
            recommendations.append("Check error logs and system health")
            recommendations.append("Consider session recovery or rollback")
        
        if 'response_time' in event.data and event.data['response_time'] > 5000:
            recommendations.append("Investigate performance bottlenecks")
            recommendations.append("Check database connection and query performance")
        
        if event.event_type == 'operation' and anomaly_score > 0.8:
            recommendations.append("Monitor session for additional anomalies")
            recommendations.append("Consider user behavior analysis")
        
        return recommendations

class RealTimeAnalyticsEngine:
    """
    Main real-time analytics engine coordinating all components
    """
    
    def __init__(self, db_manager, session_manager, ml_engine=None):
        self.db_manager = db_manager
        self.session_manager = session_manager
        self.ml_engine = ml_engine
        
        # Real-time components
        self.metrics_collector = RealTimeMetricsCollector()
        self.anomaly_detector = RealTimeAnomalyDetector()
        
        # Event processing
        self.event_queue = queue.Queue(maxsize=10000)
        self.alert_queue = queue.Queue(maxsize=1000)
        self.processing_thread = None
        self.is_running = False
        
        # WebSocket connections for real-time updates
        self.websocket_clients = set()
        self.websocket_server = None
        
        # Alert callbacks
        self.alert_callbacks = []
        
        logger.info("✅ Real-time Analytics Engine initialized")
    
    def start(self):
        """Start the real-time analytics engine"""
        if self.is_running:
            logger.warning("Real-time analytics engine already running")
            return
        
        self.is_running = True
        
        # Start event processing thread
        self.processing_thread = threading.Thread(target=self._process_events_loop, daemon=True)
        self.processing_thread.start()
        
        # Start WebSocket server if available
        if WEBSOCKET_AVAILABLE:
            asyncio.create_task(self._start_websocket_server())
        
        logger.info("🚀 Real-time Analytics Engine started")
    
    def stop(self):
        """Stop the real-time analytics engine"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        if self.websocket_server:
            self.websocket_server.close()
        
        logger.info("⏹️ Real-time Analytics Engine stopped")
    
    def add_event(self, event_type: str, session_id: str, user_id: str, 
                  data: Dict[str, Any]) -> str:
        """Add new real-time event for processing"""
        event = RealTimeEvent(
            event_id=f"{event_type}_{int(time.time())}_{hash(session_id) % 1000}",
            event_type=event_type,
            session_id=session_id,
            user_id=user_id,
            timestamp=datetime.now(),
            data=data
        )
        
        try:
            self.event_queue.put_nowait(event)
            return event.event_id
        except queue.Full:
            logger.warning("Event queue full - dropping event")
            return None
    
    def _process_events_loop(self):
        """Main event processing loop"""
        while self.is_running:
            try:
                # Get event from queue with timeout
                event = self.event_queue.get(timeout=1.0)
                
                # Process event
                self._process_single_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _process_single_event(self, event: RealTimeEvent):
        """Process a single event"""
        try:
            # Add to metrics collector
            self.metrics_collector.add_event(event)
            
            # Check for anomalies
            alert = self.anomaly_detector.detect_anomaly(event)
            if alert:
                self.alert_queue.put_nowait(alert)
                self._trigger_alert_callbacks(alert)
            
            # Send real-time updates to WebSocket clients
            self._broadcast_event_update(event)
            
            # ML-enhanced processing if available
            if self.ml_engine:
                self._apply_ml_analysis(event)
            
        except Exception as e:
            logger.error(f"Error in single event processing: {e}")
    
    def _apply_ml_analysis(self, event: RealTimeEvent):
        """Apply ML analysis to events"""
        try:
            # Extract ML features if possible
            if hasattr(self.ml_engine, 'extract_event_features'):
                features = self.ml_engine.extract_event_features(event.data)
                event.ml_features = features
            
            # Real-time predictions if session context available
            if event.session_id and hasattr(self.ml_engine, 'predict_session_outcome'):
                prediction = self.ml_engine.predict_session_outcome(event.session_id)
                if prediction:
                    event.data['ml_prediction'] = prediction
            
        except Exception as e:
            logger.warning(f"ML analysis failed for event: {e}")
    
    def _trigger_alert_callbacks(self, alert: RealTimeAlert):
        """Trigger registered alert callbacks"""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time updates"""
        if not WEBSOCKET_AVAILABLE:
            logger.warning("WebSocket not available - skipping real-time server")
            return
        
        try:
            async def handle_websocket(websocket, path):
                self.websocket_clients.add(websocket)
                try:
                    await websocket.wait_closed()
                finally:
                    self.websocket_clients.discard(websocket)
            
            self.websocket_server = await websockets.serve(
                handle_websocket, "0.0.0.0", 8765
            )
            logger.info("🌐 WebSocket server started on port 8765")
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    def _broadcast_event_update(self, event: RealTimeEvent):
        """Broadcast event updates to WebSocket clients"""
        if not self.websocket_clients:
            return
        
        try:
            update_data = {
                'type': 'event_update',
                'event': {
                    'event_id': event.event_id,
                    'event_type': event.event_type,
                    'session_id': event.session_id,
                    'timestamp': event.timestamp.isoformat(),
                    'anomaly_score': event.anomaly_score,
                    'data': event.data
                },
                'metrics': self.metrics_collector.get_current_metrics()
            }
            
            message = json.dumps(update_data)
            
            # Send to all connected clients
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    asyncio.create_task(client.send(message))
                except Exception:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.websocket_clients -= disconnected_clients
            
        except Exception as e:
            logger.warning(f"Failed to broadcast update: {e}")
    
    def register_alert_callback(self, callback: Callable[[RealTimeAlert], None]):
        """Register callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current real-time analytics status"""
        return {
            'engine_status': 'running' if self.is_running else 'stopped',
            'metrics': self.metrics_collector.get_current_metrics(),
            'event_queue_size': self.event_queue.qsize(),
            'alert_queue_size': self.alert_queue.qsize(),
            'websocket_clients': len(self.websocket_clients),
            'anomaly_detector_trained': self.anomaly_detector.is_trained,
            'recent_anomalies': len(self.anomaly_detector.recent_anomalies),
            'ml_enhanced': self.ml_engine is not None
        }
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        alerts = []
        temp_queue = []
        
        # Extract alerts from queue
        try:
            while len(alerts) < limit and not self.alert_queue.empty():
                alert = self.alert_queue.get_nowait()
                alerts.append(asdict(alert))
                temp_queue.append(alert)
        except queue.Empty:
            pass
        
        # Put alerts back in queue
        for alert in temp_queue:
            try:
                self.alert_queue.put_nowait(alert)
            except queue.Full:
                break
        
        return alerts