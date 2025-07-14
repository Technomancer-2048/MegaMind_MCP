#!/usr/bin/env python3
"""
Phase 7: Real-time Analytics MCP Server
Integrates all Phase 7 real-time components with the MCP protocol
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading

# Import Phase 6 ML-enhanced server as base
from phase6_ml_enhanced_server import Phase6MLEnhancedMCPServer

# Import Phase 7 real-time components
from realtime_analytics_engine import RealTimeAnalyticsEngine
from realtime_session_monitor import RealTimeSessionMonitor
from ml_performance_tracker import MLModelPerformanceTracker
from realtime_alert_system import RealTimeAlertSystem, AlertSeverity, AlertRule

logger = logging.getLogger(__name__)

class Phase7RealTimeMCPServer(Phase6MLEnhancedMCPServer):
    """
    Phase 7 Real-time Analytics MCP Server
    Extends Phase 6 with comprehensive real-time monitoring and alerting
    """
    
    def __init__(self, db_manager):
        # Initialize Phase 6 base
        super().__init__(db_manager)
        
        # Phase 7 real-time components
        self.realtime_analytics = None
        self.session_monitor = None
        self.ml_performance_tracker = None
        self.alert_system = None
        
        # Real-time status
        self.realtime_services_started = False
        self.websocket_servers_running = False
        
        # Initialize Phase 7 components
        self._initialize_realtime_components()
        
        # Add Phase 7 MCP functions
        self._register_phase7_functions()
        
        logger.info("✅ Phase 7 Real-time MCP Server initialized")
    
    def _initialize_realtime_components(self):
        """Initialize Phase 7 real-time components"""
        try:
            # Real-time analytics engine
            self.realtime_analytics = RealTimeAnalyticsEngine(
                db_manager=self.db_manager,
                session_manager=self.session_manager,
                ml_engine=self.ml_engine
            )
            
            # Real-time session monitor
            self.session_monitor = RealTimeSessionMonitor(
                session_manager=self.session_manager,
                ml_engine=self.ml_engine,
                analytics_engine=self.realtime_analytics
            )
            
            # ML performance tracker
            self.ml_performance_tracker = MLModelPerformanceTracker(
                ml_engine=self.ml_engine
            )
            
            # Real-time alert system
            self.alert_system = RealTimeAlertSystem(
                session_monitor=self.session_monitor,
                ml_performance_tracker=self.ml_performance_tracker,
                analytics_engine=self.realtime_analytics
            )
            
            # Register ML models for tracking
            self._register_ml_models_for_tracking()
            
            # Setup alert integrations
            self._setup_alert_integrations()
            
            logger.info("✅ Phase 7 real-time components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Phase 7 components: {e}")
            # Graceful fallback - continue with Phase 6 functionality
    
    def _register_ml_models_for_tracking(self):
        """Register ML models with the performance tracker"""
        if not self.ml_performance_tracker:
            return
        
        try:
            # Register Phase 6 ML models
            models_to_track = [
                ("session_success_predictor", "classifier", "1.0"),
                ("session_similarity_calculator", "similarity", "1.0"),
                ("session_clustering_engine", "clustering", "1.0"),
                ("topic_modeling_engine", "topic", "1.0"),
                ("anomaly_detector", "anomaly", "1.0")
            ]
            
            for model_name, model_type, version in models_to_track:
                self.ml_performance_tracker.register_model(
                    model_name=model_name,
                    model_type=model_type,
                    version=version,
                    metadata={"integrated_with": "phase7_realtime"}
                )
            
            logger.info("✅ ML models registered for performance tracking")
            
        except Exception as e:
            logger.warning(f"Failed to register ML models: {e}")
    
    def _setup_alert_integrations(self):
        """Setup alert system integrations"""
        if not self.alert_system:
            return
        
        try:
            # Register alert callback with real-time analytics
            if self.realtime_analytics:
                self.realtime_analytics.register_alert_callback(self._handle_analytics_alert)
            
            # Setup custom alert rules
            custom_rules = [
                AlertRule(
                    rule_id="phase7_ml_drift",
                    name="Phase 7 ML Model Drift",
                    description="Significant drift detected in ML model performance",
                    metric_name="model_drift",
                    condition="greater_than",
                    threshold_value=0.15,
                    severity=AlertSeverity.HIGH,
                    cooldown_minutes=30
                ),
                AlertRule(
                    rule_id="phase7_session_anomaly_cluster",
                    name="Session Anomaly Cluster",
                    description="Multiple anomalous sessions detected in short time",
                    metric_name="anomaly_cluster_size",
                    condition="greater_than",
                    threshold_value=5,
                    severity=AlertSeverity.CRITICAL,
                    cooldown_minutes=15
                )
            ]
            
            for rule in custom_rules:
                self.alert_system.add_alert_rule(rule)
            
            logger.info("✅ Alert system integrations configured")
            
        except Exception as e:
            logger.warning(f"Failed to setup alert integrations: {e}")
    
    def _handle_analytics_alert(self, alert):
        """Handle alerts from real-time analytics engine"""
        try:
            # Convert analytics alert to system alert
            asyncio.create_task(self.alert_system.create_alert(
                rule_id="realtime_analytics_alert",
                metric_name=alert.source_event.event_type,
                current_value=alert.threshold_violated.get('anomaly_score', 1.0),
                source="realtime_analytics",
                context={
                    "event_id": alert.source_event.event_id,
                    "session_id": alert.source_event.session_id,
                    "original_alert_type": alert.alert_type
                },
                custom_message=alert.message
            ))
        except Exception as e:
            logger.error(f"Failed to handle analytics alert: {e}")
    
    def _register_phase7_functions(self):
        """Register Phase 7 MCP functions"""
        phase7_functions = {
            # Real-time monitoring functions
            "mcp__megamind__realtime_start_monitoring": self.handle_realtime_start_monitoring,
            "mcp__megamind__realtime_stop_monitoring": self.handle_realtime_stop_monitoring,
            "mcp__megamind__realtime_get_status": self.handle_realtime_get_status,
            "mcp__megamind__realtime_subscribe_session": self.handle_realtime_subscribe_session,
            
            # Real-time analytics functions
            "mcp__megamind__realtime_add_event": self.handle_realtime_add_event,
            "mcp__megamind__realtime_get_metrics": self.handle_realtime_get_metrics,
            "mcp__megamind__realtime_get_analytics": self.handle_realtime_get_analytics,
            
            # ML performance tracking functions
            "mcp__megamind__ml_performance_status": self.handle_ml_performance_status,
            "mcp__megamind__ml_record_prediction": self.handle_ml_record_prediction,
            "mcp__megamind__ml_record_outcome": self.handle_ml_record_outcome,
            "mcp__megamind__ml_get_performance_metrics": self.handle_ml_get_performance_metrics,
            
            # Alert system functions
            "mcp__megamind__alerts_get_active": self.handle_alerts_get_active,
            "mcp__megamind__alerts_acknowledge": self.handle_alerts_acknowledge,
            "mcp__megamind__alerts_resolve": self.handle_alerts_resolve,
            "mcp__megamind__alerts_create_custom": self.handle_alerts_create_custom,
            "mcp__megamind__alerts_get_metrics": self.handle_alerts_get_metrics,
            
            # Streaming dashboard functions
            "mcp__megamind__dashboard_realtime": self.handle_dashboard_realtime,
            "mcp__megamind__dashboard_get_connections": self.handle_dashboard_get_connections
        }
        
        # Add to function registry
        for name, handler in phase7_functions.items():
            self.mcp_functions[name] = handler
        
        logger.info(f"✅ Registered {len(phase7_functions)} Phase 7 MCP functions")
    
    async def start_realtime_services(self):
        """Start all real-time services"""
        if self.realtime_services_started:
            return True
        
        try:
            # Start real-time analytics
            if self.realtime_analytics:
                self.realtime_analytics.start()
            
            # Start session monitor WebSocket server
            if self.session_monitor:
                await self.session_monitor.start_server()
                self.websocket_servers_running = True
            
            # Start ML performance tracking
            if self.ml_performance_tracker:
                self.ml_performance_tracker.start_tracking()
            
            # Start alert system
            if self.alert_system:
                await self.alert_system.start()
            
            self.realtime_services_started = True
            logger.info("🚀 All Phase 7 real-time services started")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start real-time services: {e}")
            return False
    
    async def stop_realtime_services(self):
        """Stop all real-time services"""
        if not self.realtime_services_started:
            return
        
        try:
            # Stop alert system
            if self.alert_system:
                await self.alert_system.stop()
            
            # Stop ML performance tracking
            if self.ml_performance_tracker:
                self.ml_performance_tracker.stop_tracking()
            
            # Stop session monitor
            if self.session_monitor:
                await self.session_monitor.stop_server()
                self.websocket_servers_running = False
            
            # Stop real-time analytics
            if self.realtime_analytics:
                self.realtime_analytics.stop()
            
            self.realtime_services_started = False
            logger.info("⏹️ All Phase 7 real-time services stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping real-time services: {e}")
    
    # ================================================================
    # PHASE 7 MCP FUNCTION HANDLERS
    # ================================================================
    
    def handle_realtime_start_monitoring(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Start real-time monitoring services"""
        try:
            # Start services asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(self.start_realtime_services())
            
            if success:
                return {
                    "success": True,
                    "message": "Real-time monitoring services started",
                    "services_status": self._get_services_status(),
                    "websocket_endpoints": {
                        "session_monitor": f"ws://localhost:{getattr(self.session_monitor, 'server_port', 8766)}",
                        "analytics_stream": f"ws://localhost:8765"
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to start some real-time services",
                    "services_status": self._get_services_status()
                }
                
        except Exception as e:
            logger.error(f"Error starting real-time monitoring: {e}")
            return {
                "success": False,
                "error": f"Failed to start real-time monitoring: {str(e)}"
            }
    
    def handle_realtime_stop_monitoring(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Stop real-time monitoring services"""
        try:
            # Stop services asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.stop_realtime_services())
            
            return {
                "success": True,
                "message": "Real-time monitoring services stopped",
                "services_status": self._get_services_status()
            }
            
        except Exception as e:
            logger.error(f"Error stopping real-time monitoring: {e}")
            return {
                "success": False,
                "error": f"Failed to stop real-time monitoring: {str(e)}"
            }
    
    def handle_realtime_get_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time monitoring status"""
        try:
            return {
                "success": True,
                "realtime_status": {
                    "services_started": self.realtime_services_started,
                    "websocket_servers_running": self.websocket_servers_running,
                    "services": self._get_services_status(),
                    "phase7_capabilities": self._get_phase7_capabilities()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting realtime status: {e}")
            return {
                "success": False,
                "error": f"Failed to get status: {str(e)}"
            }
    
    def handle_realtime_subscribe_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Subscribe to real-time session updates"""
        session_id = args.get('session_id')
        client_id = args.get('client_id')
        
        if not session_id:
            return {"success": False, "error": "session_id required"}
        
        try:
            # This would typically be handled by WebSocket connections
            # For MCP, we return subscription instructions
            return {
                "success": True,
                "message": f"Subscription instructions for session {session_id}",
                "websocket_url": f"ws://localhost:{getattr(self.session_monitor, 'server_port', 8766)}",
                "subscription_message": {
                    "type": "subscribe_session",
                    "session_id": session_id,
                    "client_id": client_id
                },
                "available_updates": [
                    "status_change", "operation", "performance", "error", "completion"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error setting up session subscription: {e}")
            return {
                "success": False,
                "error": f"Failed to setup subscription: {str(e)}"
            }
    
    def handle_realtime_add_event(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Add real-time event to analytics"""
        event_type = args.get('event_type')
        session_id = args.get('session_id')
        user_id = args.get('user_id')
        data = args.get('data', {})
        
        if not all([event_type, session_id, user_id]):
            return {"success": False, "error": "event_type, session_id, and user_id required"}
        
        try:
            if self.realtime_analytics:
                event_id = self.realtime_analytics.add_event(event_type, session_id, user_id, data)
                
                # Also broadcast to session monitor if available
                if self.session_monitor:
                    asyncio.create_task(
                        self.session_monitor.broadcast_session_update(
                            session_id, event_type, data, user_id
                        )
                    )
                
                return {
                    "success": True,
                    "event_id": event_id,
                    "message": "Event added to real-time analytics",
                    "queued_for_processing": True
                }
            else:
                return {
                    "success": False,
                    "error": "Real-time analytics not available"
                }
                
        except Exception as e:
            logger.error(f"Error adding real-time event: {e}")
            return {
                "success": False,
                "error": f"Failed to add event: {str(e)}"
            }
    
    def handle_realtime_get_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time analytics metrics"""
        try:
            if not self.realtime_analytics:
                return {"success": False, "error": "Real-time analytics not available"}
            
            current_metrics = self.realtime_analytics.metrics_collector.get_current_metrics()
            
            return {
                "success": True,
                "realtime_metrics": current_metrics,
                "analytics_status": self.realtime_analytics.get_current_status()
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time metrics: {e}")
            return {
                "success": False,
                "error": f"Failed to get metrics: {str(e)}"
            }
    
    def handle_realtime_get_analytics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive real-time analytics"""
        time_window = args.get('time_window', '1h')
        include_anomalies = args.get('include_anomalies', True)
        
        try:
            analytics_data = {
                "timestamp": datetime.now().isoformat(),
                "time_window": time_window
            }
            
            # Get real-time metrics
            if self.realtime_analytics:
                analytics_data["current_metrics"] = self.realtime_analytics.get_current_status()
            
            # Get ML performance data
            if self.ml_performance_tracker:
                analytics_data["ml_performance"] = self.ml_performance_tracker.get_all_model_performance()
            
            # Get recent alerts
            if self.alert_system and include_anomalies:
                hours = 1 if time_window == '1h' else 24 if time_window == '24h' else 1
                analytics_data["recent_alerts"] = self.alert_system.get_alert_history(hours=hours)
            
            # Get session monitoring data
            if self.session_monitor:
                analytics_data["session_monitoring"] = self.session_monitor.get_monitoring_status()
            
            return {
                "success": True,
                "realtime_analytics": analytics_data
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time analytics: {e}")
            return {
                "success": False,
                "error": f"Failed to get analytics: {str(e)}"
            }
    
    def handle_ml_performance_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML performance tracking status"""
        try:
            if not self.ml_performance_tracker:
                return {"success": False, "error": "ML performance tracker not available"}
            
            status = self.ml_performance_tracker.get_tracking_status()
            
            return {
                "success": True,
                "ml_performance_tracking": status
            }
            
        except Exception as e:
            logger.error(f"Error getting ML performance status: {e}")
            return {
                "success": False,
                "error": f"Failed to get ML performance status: {str(e)}"
            }
    
    def handle_ml_record_prediction(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Record ML model prediction for performance tracking"""
        model_name = args.get('model_name')
        input_features = args.get('input_features', {})
        prediction = args.get('prediction')
        confidence = args.get('confidence')
        prediction_time_ms = args.get('prediction_time_ms')
        
        if not model_name or prediction is None:
            return {"success": False, "error": "model_name and prediction required"}
        
        try:
            if not self.ml_performance_tracker:
                return {"success": False, "error": "ML performance tracker not available"}
            
            prediction_id = self.ml_performance_tracker.record_prediction(
                model_name=model_name,
                input_features=input_features,
                prediction=prediction,
                confidence=confidence,
                prediction_time_ms=prediction_time_ms
            )
            
            return {
                "success": True,
                "prediction_id": prediction_id,
                "message": "Prediction recorded for performance tracking"
            }
            
        except Exception as e:
            logger.error(f"Error recording ML prediction: {e}")
            return {
                "success": False,
                "error": f"Failed to record prediction: {str(e)}"
            }
    
    def handle_ml_record_outcome(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Record actual outcome for ML prediction"""
        prediction_id = args.get('prediction_id')
        actual_outcome = args.get('actual_outcome')
        
        if not prediction_id or actual_outcome is None:
            return {"success": False, "error": "prediction_id and actual_outcome required"}
        
        try:
            if not self.ml_performance_tracker:
                return {"success": False, "error": "ML performance tracker not available"}
            
            self.ml_performance_tracker.record_actual_outcome(prediction_id, actual_outcome)
            
            return {
                "success": True,
                "message": "Actual outcome recorded, performance metrics updated"
            }
            
        except Exception as e:
            logger.error(f"Error recording ML outcome: {e}")
            return {
                "success": False,
                "error": f"Failed to record outcome: {str(e)}"
            }
    
    def handle_ml_get_performance_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get ML model performance metrics"""
        model_name = args.get('model_name')
        include_history = args.get('include_history', False)
        hours = args.get('hours', 24)
        
        try:
            if not self.ml_performance_tracker:
                return {"success": False, "error": "ML performance tracker not available"}
            
            if model_name:
                # Get specific model metrics
                current_metrics = self.ml_performance_tracker.get_model_performance(model_name)
                if not current_metrics:
                    return {"success": False, "error": f"No metrics found for model {model_name}"}
                
                result = {
                    "success": True,
                    "model_name": model_name,
                    "current_metrics": current_metrics
                }
                
                if include_history:
                    history = self.ml_performance_tracker.get_performance_history(model_name, hours)
                    result["performance_history"] = history
                
                return result
            else:
                # Get all model metrics
                all_metrics = self.ml_performance_tracker.get_all_model_performance()
                return {
                    "success": True,
                    "all_models_metrics": all_metrics
                }
                
        except Exception as e:
            logger.error(f"Error getting ML performance metrics: {e}")
            return {
                "success": False,
                "error": f"Failed to get performance metrics: {str(e)}"
            }
    
    def handle_alerts_get_active(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get active alerts"""
        severity_filter = args.get('severity_filter')
        
        try:
            if not self.alert_system:
                return {"success": False, "error": "Alert system not available"}
            
            # Convert severity filter if provided
            severity_enum_filter = None
            if severity_filter:
                severity_enum_filter = [AlertSeverity(s) for s in severity_filter]
            
            active_alerts = self.alert_system.get_active_alerts(severity_enum_filter)
            
            return {
                "success": True,
                "active_alerts": active_alerts,
                "total_active": len(active_alerts)
            }
            
        except Exception as e:
            logger.error(f"Error getting active alerts: {e}")
            return {
                "success": False,
                "error": f"Failed to get active alerts: {str(e)}"
            }
    
    def handle_alerts_acknowledge(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Acknowledge an alert"""
        alert_id = args.get('alert_id')
        acknowledged_by = args.get('acknowledged_by', 'user')
        
        if not alert_id:
            return {"success": False, "error": "alert_id required"}
        
        try:
            if not self.alert_system:
                return {"success": False, "error": "Alert system not available"}
            
            # Run async acknowledge method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(
                self.alert_system.acknowledge_alert(alert_id, acknowledged_by)
            )
            
            if success:
                return {
                    "success": True,
                    "message": f"Alert {alert_id} acknowledged by {acknowledged_by}"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to acknowledge alert {alert_id} - may not exist or already acknowledged"
                }
                
        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return {
                "success": False,
                "error": f"Failed to acknowledge alert: {str(e)}"
            }
    
    def handle_alerts_resolve(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve an alert"""
        alert_id = args.get('alert_id')
        resolved_by = args.get('resolved_by', 'user')
        
        if not alert_id:
            return {"success": False, "error": "alert_id required"}
        
        try:
            if not self.alert_system:
                return {"success": False, "error": "Alert system not available"}
            
            # Run async resolve method
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(
                self.alert_system.resolve_alert(alert_id, resolved_by)
            )
            
            if success:
                return {
                    "success": True,
                    "message": f"Alert {alert_id} resolved"
                }
            else:
                return {
                    "success": False,
                    "error": f"Failed to resolve alert {alert_id} - may not exist"
                }
                
        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return {
                "success": False,
                "error": f"Failed to resolve alert: {str(e)}"
            }
    
    def handle_alerts_create_custom(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Create custom alert"""
        rule_id = args.get('rule_id', 'custom_alert')
        metric_name = args.get('metric_name')
        current_value = args.get('current_value')
        source = args.get('source', 'manual')
        context = args.get('context', {})
        message = args.get('message')
        
        if not all([metric_name, current_value is not None]):
            return {"success": False, "error": "metric_name and current_value required"}
        
        try:
            if not self.alert_system:
                return {"success": False, "error": "Alert system not available"}
            
            # Create custom alert
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            alert_id = loop.run_until_complete(
                self.alert_system.create_alert(
                    rule_id=rule_id,
                    metric_name=metric_name,
                    current_value=float(current_value),
                    source=source,
                    context=context,
                    custom_message=message
                )
            )
            
            if alert_id:
                return {
                    "success": True,
                    "alert_id": alert_id,
                    "message": "Custom alert created"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create custom alert - may be in cooldown or rule disabled"
                }
                
        except Exception as e:
            logger.error(f"Error creating custom alert: {e}")
            return {
                "success": False,
                "error": f"Failed to create custom alert: {str(e)}"
            }
    
    def handle_alerts_get_metrics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get alert system metrics"""
        try:
            if not self.alert_system:
                return {"success": False, "error": "Alert system not available"}
            
            metrics = self.alert_system.get_alert_metrics()
            system_status = self.alert_system.get_system_status()
            
            return {
                "success": True,
                "alert_metrics": metrics,
                "system_status": system_status
            }
            
        except Exception as e:
            logger.error(f"Error getting alert metrics: {e}")
            return {
                "success": False,
                "error": f"Failed to get alert metrics: {str(e)}"
            }
    
    def handle_dashboard_realtime(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time dashboard data"""
        user_id = args.get('user_id')
        realm_id = args.get('realm_id', 'MegaMind_MCP')
        include_predictions = args.get('include_predictions', True)
        
        try:
            dashboard_data = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "realm_id": realm_id
            }
            
            # Get real-time session data
            if self.session_monitor:
                dashboard_data["session_monitoring"] = self.session_monitor.get_monitoring_status()
                dashboard_data["connected_clients"] = self.session_monitor.get_connected_clients()
            
            # Get real-time analytics
            if self.realtime_analytics:
                dashboard_data["analytics"] = self.realtime_analytics.get_current_status()
            
            # Get ML performance data
            if self.ml_performance_tracker and include_predictions:
                dashboard_data["ml_performance"] = self.ml_performance_tracker.get_all_model_performance()
            
            # Get active alerts
            if self.alert_system:
                dashboard_data["active_alerts"] = self.alert_system.get_active_alerts()
                dashboard_data["alert_metrics"] = self.alert_system.get_alert_metrics()
            
            # Add Phase 6 ML insights if available
            if self.ml_engine_available:
                dashboard_data["ml_insights_available"] = True
                # Could add real-time ML predictions here
            
            return {
                "success": True,
                "realtime_dashboard": dashboard_data,
                "refresh_rate_seconds": 5,  # Recommended refresh rate
                "websocket_available": self.websocket_servers_running
            }
            
        except Exception as e:
            logger.error(f"Error getting real-time dashboard: {e}")
            return {
                "success": False,
                "error": f"Failed to get real-time dashboard: {str(e)}"
            }
    
    def handle_dashboard_get_connections(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Get WebSocket connection information"""
        try:
            connections = {
                "websocket_servers_running": self.websocket_servers_running,
                "available_endpoints": {}
            }
            
            if self.session_monitor:
                connections["available_endpoints"]["session_monitor"] = {
                    "url": f"ws://localhost:{getattr(self.session_monitor, 'server_port', 8766)}",
                    "description": "Real-time session monitoring and updates",
                    "message_types": ["subscribe_session", "get_session_status", "get_performance_metrics"]
                }
            
            if self.realtime_analytics:
                connections["available_endpoints"]["analytics_stream"] = {
                    "url": "ws://localhost:8765",
                    "description": "Real-time analytics and event stream",
                    "message_types": ["event_update", "metrics_update", "anomaly_alert"]
                }
            
            return {
                "success": True,
                "websocket_connections": connections
            }
            
        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            return {
                "success": False,
                "error": f"Failed to get connection info: {str(e)}"
            }
    
    # ================================================================
    # HELPER METHODS
    # ================================================================
    
    def _get_services_status(self) -> Dict[str, Any]:
        """Get status of all Phase 7 services"""
        return {
            "realtime_analytics": {
                "initialized": self.realtime_analytics is not None,
                "running": self.realtime_analytics.is_running if self.realtime_analytics else False
            },
            "session_monitor": {
                "initialized": self.session_monitor is not None,
                "server_running": getattr(self.session_monitor, 'is_running', False)
            },
            "ml_performance_tracker": {
                "initialized": self.ml_performance_tracker is not None,
                "tracking_active": getattr(self.ml_performance_tracker, 'is_running', False)
            },
            "alert_system": {
                "initialized": self.alert_system is not None,
                "running": getattr(self.alert_system, 'is_running', False)
            }
        }
    
    def _get_phase7_capabilities(self) -> Dict[str, Any]:
        """Get Phase 7 specific capabilities"""
        return {
            "real_time_monitoring": True,
            "websocket_streaming": WEBSOCKET_AVAILABLE,
            "ml_performance_tracking": self.ml_performance_tracker is not None,
            "anomaly_detection": self.realtime_analytics is not None,
            "alert_system": self.alert_system is not None,
            "session_streaming": self.session_monitor is not None,
            "predictive_insights": self.ml_engine_available,
            "multi_channel_alerts": True,
            "real_time_dashboard": True
        }
    
    def get_phase7_status(self) -> Dict[str, Any]:
        """Get comprehensive Phase 7 status"""
        # Get Phase 6 status as base
        phase6_status = self.get_phase6_status()
        
        # Add Phase 7 specific status
        phase7_status = {
            "phase7_realtime_available": True,
            "realtime_services_started": self.realtime_services_started,
            "websocket_servers_running": self.websocket_servers_running,
            "services_status": self._get_services_status(),
            "phase7_capabilities": self._get_phase7_capabilities(),
            "total_phase7_functions": len([f for f in self.mcp_functions.keys() 
                                         if f.startswith("mcp__megamind__realtime_") or
                                            f.startswith("mcp__megamind__ml_") or
                                            f.startswith("mcp__megamind__alerts_") or
                                            f.startswith("mcp__megamind__dashboard_")])
        }
        
        # Merge with Phase 6 status
        return {**phase6_status, **phase7_status}