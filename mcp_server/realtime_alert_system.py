#!/usr/bin/env python3
"""
Phase 7: Real-time Alert System
Comprehensive alerting system with immediate notifications and escalation
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
from enum import Enum
import uuid
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart

# WebSocket for real-time notifications
try:
    import websockets
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals', 'threshold_breach'
    threshold_value: float
    severity: AlertSeverity
    cooldown_minutes: int = 5
    escalation_minutes: int = 30
    enabled: bool = True
    tags: List[str] = None

@dataclass
class Alert:
    """Real-time alert data structure"""
    alert_id: str
    rule_id: str
    title: str
    message: str
    severity: AlertSeverity
    status: AlertStatus
    source: str  # 'ml_performance', 'session_monitor', 'system_health', 'anomaly_detection'
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    tags: List[str] = None
    context: Dict[str, Any] = None
    escalated: bool = False
    escalated_at: Optional[datetime] = None

@dataclass
class NotificationChannel:
    """Notification channel configuration"""
    channel_id: str
    channel_type: str  # 'email', 'webhook', 'websocket', 'slack', 'teams'
    name: str
    config: Dict[str, Any]
    severity_filter: List[AlertSeverity]
    enabled: bool = True

class AlertEscalationManager:
    """
    Manages alert escalation and notification routing
    """
    
    def __init__(self):
        self.escalation_rules = {}  # rule_id -> escalation config
        self.notification_channels = {}  # channel_id -> NotificationChannel
        self.escalation_history = deque(maxlen=1000)
        
    def add_escalation_rule(self, rule_id: str, escalation_config: Dict[str, Any]):
        """Add escalation rule for specific alert rule"""
        self.escalation_rules[rule_id] = escalation_config
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.notification_channels[channel.channel_id] = channel
    
    def should_escalate(self, alert: Alert) -> bool:
        """Check if alert should be escalated"""
        if alert.escalated or alert.status != AlertStatus.ACTIVE:
            return False
        
        escalation_config = self.escalation_rules.get(alert.rule_id)
        if not escalation_config:
            return False
        
        escalation_minutes = escalation_config.get('escalation_minutes', 30)
        time_since_alert = datetime.now() - alert.timestamp
        
        return time_since_alert.total_seconds() >= escalation_minutes * 60
    
    def get_notification_channels(self, alert: Alert) -> List[NotificationChannel]:
        """Get appropriate notification channels for alert"""
        channels = []
        
        for channel in self.notification_channels.values():
            if not channel.enabled:
                continue
            
            # Check severity filter
            if alert.severity in channel.severity_filter:
                channels.append(channel)
        
        return channels

class RealTimeAlertSystem:
    """
    Comprehensive real-time alerting system with multiple notification channels
    """
    
    def __init__(self, session_monitor=None, ml_performance_tracker=None, analytics_engine=None):
        self.session_monitor = session_monitor
        self.ml_performance_tracker = ml_performance_tracker
        self.analytics_engine = analytics_engine
        
        # Alert management
        self.active_alerts = {}  # alert_id -> Alert
        self.alert_history = deque(maxlen=10000)
        self.alert_rules = {}  # rule_id -> AlertRule
        
        # Escalation and notification
        self.escalation_manager = AlertEscalationManager()
        
        # Alert processing
        self.alert_queue = asyncio.Queue(maxsize=1000)
        self.is_running = False
        self.processing_task = None
        
        # Metrics tracking
        self.alert_metrics = {
            'total_alerts': 0,
            'alerts_by_severity': defaultdict(int),
            'alerts_by_source': defaultdict(int),
            'avg_resolution_time': 0.0,
            'escalation_rate': 0.0
        }
        
        # Rate limiting and cooldowns
        self.cooldown_tracker = {}  # rule_id -> last_alert_time
        self.rate_limits = defaultdict(lambda: deque(maxlen=100))  # rule_id -> timestamps
        
        self._setup_default_rules()
        self._setup_default_channels()
        
        logger.info("✅ Real-time Alert System initialized")
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                rule_id="ml_accuracy_drop",
                name="ML Model Accuracy Drop",
                description="ML model accuracy below threshold",
                metric_name="accuracy",
                condition="less_than",
                threshold_value=0.8,
                severity=AlertSeverity.HIGH,
                cooldown_minutes=10,
                escalation_minutes=30
            ),
            AlertRule(
                rule_id="high_latency",
                name="High Response Latency",
                description="Response time exceeds threshold",
                metric_name="response_time",
                condition="greater_than",
                threshold_value=5000,  # 5 seconds
                severity=AlertSeverity.MEDIUM,
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="session_error_rate",
                name="High Session Error Rate",
                description="Session error rate above threshold",
                metric_name="error_rate",
                condition="greater_than",
                threshold_value=0.1,  # 10%
                severity=AlertSeverity.HIGH,
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="anomaly_detected",
                name="Anomaly Detected",
                description="System anomaly detected",
                metric_name="anomaly_score",
                condition="greater_than",
                threshold_value=0.7,
                severity=AlertSeverity.MEDIUM,
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="system_resource_critical",
                name="Critical System Resources",
                description="System resources critically low",
                metric_name="resource_usage",
                condition="greater_than",
                threshold_value=0.9,  # 90%
                severity=AlertSeverity.CRITICAL,
                cooldown_minutes=5,
                escalation_minutes=15
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
    
    def _setup_default_channels(self):
        """Setup default notification channels"""
        # WebSocket channel for real-time dashboard updates
        if WEBSOCKET_AVAILABLE:
            websocket_channel = NotificationChannel(
                channel_id="websocket_dashboard",
                channel_type="websocket",
                name="Real-time Dashboard",
                config={"port": 8767},
                severity_filter=[AlertSeverity.LOW, AlertSeverity.MEDIUM, 
                               AlertSeverity.HIGH, AlertSeverity.CRITICAL]
            )
            self.escalation_manager.add_notification_channel(websocket_channel)
        
        # Log channel for all alerts
        log_channel = NotificationChannel(
            channel_id="log_channel",
            channel_type="log",
            name="System Logs",
            config={"log_level": "WARNING"},
            severity_filter=[AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
        )
        self.escalation_manager.add_notification_channel(log_channel)
    
    async def start(self):
        """Start the alert system"""
        if self.is_running:
            logger.warning("⚠️ Alert system already running")
            return
        
        self.is_running = True
        
        # Start alert processing
        self.processing_task = asyncio.create_task(self._process_alerts_loop())
        
        # Start escalation monitoring
        asyncio.create_task(self._escalation_monitoring_loop())
        
        # Start metrics calculation
        asyncio.create_task(self._metrics_calculation_loop())
        
        logger.info("🚀 Real-time Alert System started")
    
    async def stop(self):
        """Stop the alert system"""
        self.is_running = False
        
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        logger.info("⏹️ Real-time Alert System stopped")
    
    async def create_alert(self, rule_id: str, metric_name: str, current_value: float,
                          source: str, context: Dict[str, Any] = None, 
                          custom_message: str = None) -> Optional[str]:
        """Create a new alert"""
        rule = self.alert_rules.get(rule_id)
        if not rule or not rule.enabled:
            return None
        
        # Check cooldown
        if self._is_in_cooldown(rule_id):
            return None
        
        # Check rate limiting
        if self._is_rate_limited(rule_id):
            logger.warning(f"Rate limit exceeded for rule {rule_id}")
            return None
        
        # Check if alert condition is met
        if not self._evaluate_condition(rule, current_value):
            return None
        
        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule_id=rule_id,
            title=rule.name,
            message=custom_message or rule.description,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            source=source,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=rule.threshold_value,
            timestamp=datetime.now(),
            tags=rule.tags or [],
            context=context or {}
        )
        
        # Queue alert for processing
        try:
            await self.alert_queue.put(alert)
            self.cooldown_tracker[rule_id] = datetime.now()
            self._update_rate_limit(rule_id)
            
            return alert.alert_id
        except asyncio.QueueFull:
            logger.error("Alert queue full - dropping alert")
            return None
    
    def _evaluate_condition(self, rule: AlertRule, current_value: float) -> bool:
        """Evaluate if alert condition is met"""
        threshold = rule.threshold_value
        
        if rule.condition == "greater_than":
            return current_value > threshold
        elif rule.condition == "less_than":
            return current_value < threshold
        elif rule.condition == "equals":
            return abs(current_value - threshold) < 0.001
        elif rule.condition == "not_equals":
            return abs(current_value - threshold) >= 0.001
        elif rule.condition == "threshold_breach":
            return abs(current_value - threshold) > threshold * 0.1
        
        return False
    
    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if rule is in cooldown period"""
        last_alert = self.cooldown_tracker.get(rule_id)
        if not last_alert:
            return False
        
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
        
        cooldown_end = last_alert + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def _is_rate_limited(self, rule_id: str) -> bool:
        """Check if rule is rate limited"""
        timestamps = self.rate_limits[rule_id]
        
        # Allow max 10 alerts per hour per rule
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_alerts = [ts for ts in timestamps if ts > one_hour_ago]
        
        return len(recent_alerts) >= 10
    
    def _update_rate_limit(self, rule_id: str):
        """Update rate limit tracking"""
        self.rate_limits[rule_id].append(datetime.now())
    
    async def _process_alerts_loop(self):
        """Main alert processing loop"""
        while self.is_running:
            try:
                # Get alert from queue
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                
                # Process alert
                await self._process_single_alert(alert)
                
                # Mark task done
                self.alert_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    async def _process_single_alert(self, alert: Alert):
        """Process a single alert"""
        try:
            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            
            # Update metrics
            self._update_alert_metrics(alert)
            
            # Send notifications
            await self._send_notifications(alert)
            
            logger.info(f"🚨 Alert created: {alert.title} ({alert.severity.value})")
            
        except Exception as e:
            logger.error(f"Error processing alert {alert.alert_id}: {e}")
    
    async def _send_notifications(self, alert: Alert):
        """Send notifications through appropriate channels"""
        channels = self.escalation_manager.get_notification_channels(alert)
        
        for channel in channels:
            try:
                await self._send_channel_notification(channel, alert)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.name}: {e}")
    
    async def _send_channel_notification(self, channel: NotificationChannel, alert: Alert):
        """Send notification through specific channel"""
        if channel.channel_type == "log":
            self._send_log_notification(alert)
        elif channel.channel_type == "websocket":
            await self._send_websocket_notification(channel, alert)
        elif channel.channel_type == "email":
            await self._send_email_notification(channel, alert)
        elif channel.channel_type == "webhook":
            await self._send_webhook_notification(channel, alert)
    
    def _send_log_notification(self, alert: Alert):
        """Send alert to system logs"""
        log_level = logging.WARNING
        if alert.severity == AlertSeverity.CRITICAL:
            log_level = logging.CRITICAL
        elif alert.severity == AlertSeverity.HIGH:
            log_level = logging.ERROR
        
        logger.log(log_level, f"ALERT: {alert.title} - {alert.message} "
                             f"(Value: {alert.current_value}, Threshold: {alert.threshold_value})")
    
    async def _send_websocket_notification(self, channel: NotificationChannel, alert: Alert):
        """Send alert via WebSocket"""
        if not WEBSOCKET_AVAILABLE:
            return
        
        # This would integrate with the session monitor's WebSocket connections
        if self.session_monitor and hasattr(self.session_monitor, 'websocket_clients'):
            message = {
                'type': 'alert_notification',
                'alert': asdict(alert),
                'timestamp': alert.timestamp.isoformat()
            }
            
            # Broadcast to all connected clients
            for client in self.session_monitor.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except Exception:
                    pass  # Client disconnected
    
    async def _send_email_notification(self, channel: NotificationChannel, alert: Alert):
        """Send alert via email"""
        try:
            config = channel.config
            smtp_server = config.get('smtp_server', 'localhost')
            smtp_port = config.get('smtp_port', 587)
            username = config.get('username')
            password = config.get('password')
            recipients = config.get('recipients', [])
            
            if not recipients:
                return
            
            # Create email message
            msg = MimeMultipart()
            msg['From'] = username or 'noreply@megamind-mcp.local'
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert: {alert.title}
Severity: {alert.severity.value}
Source: {alert.source}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold_value}
Time: {alert.timestamp}

Message: {alert.message}

Alert ID: {alert.alert_id}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                if username and password:
                    server.starttls()
                    server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"📧 Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
    
    async def _send_webhook_notification(self, channel: NotificationChannel, alert: Alert):
        """Send alert via webhook"""
        try:
            import aiohttp
            
            config = channel.config
            webhook_url = config.get('webhook_url')
            if not webhook_url:
                return
            
            payload = {
                'alert_id': alert.alert_id,
                'title': alert.title,
                'message': alert.message,
                'severity': alert.severity.value,
                'source': alert.source,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'timestamp': alert.timestamp.isoformat(),
                'context': alert.context
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"🌐 Webhook alert sent for {alert.alert_id}")
                    else:
                        logger.warning(f"Webhook returned status {response.status}")
        
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
    
    async def _escalation_monitoring_loop(self):
        """Monitor for alerts that need escalation"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for alert in list(self.active_alerts.values()):
                    if self.escalation_manager.should_escalate(alert):
                        await self._escalate_alert(alert)
                
                # Sleep for 60 seconds between checks
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in escalation monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _escalate_alert(self, alert: Alert):
        """Escalate an alert"""
        alert.escalated = True
        alert.escalated_at = datetime.now()
        
        # Create escalated alert with higher severity
        escalated_severity = AlertSeverity.CRITICAL if alert.severity != AlertSeverity.CRITICAL else AlertSeverity.CRITICAL
        
        escalated_alert = Alert(
            alert_id=str(uuid.uuid4()),
            rule_id=alert.rule_id,
            title=f"ESCALATED: {alert.title}",
            message=f"Alert escalated after {alert.escalated_at - alert.timestamp}: {alert.message}",
            severity=escalated_severity,
            status=AlertStatus.ACTIVE,
            source=alert.source,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value,
            timestamp=datetime.now(),
            tags=alert.tags + ['escalated'],
            context=alert.context
        )
        
        await self._process_single_alert(escalated_alert)
        
        logger.warning(f"🚨 Alert escalated: {alert.alert_id} -> {escalated_alert.alert_id}")
    
    async def _metrics_calculation_loop(self):
        """Calculate alert system metrics"""
        while self.is_running:
            try:
                await self._calculate_metrics()
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error calculating alert metrics: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_metrics(self):
        """Calculate current alert metrics"""
        # Reset counters
        self.alert_metrics['alerts_by_severity'].clear()
        self.alert_metrics['alerts_by_source'].clear()
        
        # Count alerts from last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
        
        self.alert_metrics['total_alerts'] = len(recent_alerts)
        
        for alert in recent_alerts:
            self.alert_metrics['alerts_by_severity'][alert.severity.value] += 1
            self.alert_metrics['alerts_by_source'][alert.source] += 1
        
        # Calculate resolution times
        resolved_alerts = [a for a in recent_alerts if a.resolved_at]
        if resolved_alerts:
            resolution_times = [
                (a.resolved_at - a.timestamp).total_seconds() / 60
                for a in resolved_alerts
            ]
            self.alert_metrics['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)
        
        # Calculate escalation rate
        escalated_alerts = [a for a in recent_alerts if a.escalated]
        if recent_alerts:
            self.alert_metrics['escalation_rate'] = len(escalated_alerts) / len(recent_alerts)
    
    def _update_alert_metrics(self, alert: Alert):
        """Update alert metrics when new alert is created"""
        self.alert_metrics['total_alerts'] += 1
        self.alert_metrics['alerts_by_severity'][alert.severity.value] += 1
        self.alert_metrics['alerts_by_source'][alert.source] += 1
    
    # ================================================================
    # PUBLIC API METHODS
    # ================================================================
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert"""
        alert = self.active_alerts.get(alert_id)
        if not alert or alert.status != AlertStatus.ACTIVE:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now()
        alert.acknowledged_by = acknowledged_by
        
        logger.info(f"✅ Alert acknowledged: {alert_id} by {acknowledged_by}")
        return True
    
    async def resolve_alert(self, alert_id: str, resolved_by: str = None) -> bool:
        """Resolve an alert"""
        alert = self.active_alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        # Remove from active alerts
        self.active_alerts.pop(alert_id, None)
        
        logger.info(f"✅ Alert resolved: {alert_id}")
        return True
    
    def get_active_alerts(self, severity_filter: Optional[List[AlertSeverity]] = None) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        alerts = []
        
        for alert in self.active_alerts.values():
            if severity_filter and alert.severity not in severity_filter:
                continue
            alerts.append(asdict(alert))
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_alert_history(self, hours: int = 24, 
                         severity_filter: Optional[List[AlertSeverity]] = None) -> List[Dict[str, Any]]:
        """Get alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        alerts = []
        
        for alert in self.alert_history:
            if alert.timestamp < cutoff_time:
                continue
            if severity_filter and alert.severity not in severity_filter:
                continue
            alerts.append(asdict(alert))
        
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)
    
    def get_alert_metrics(self) -> Dict[str, Any]:
        """Get current alert system metrics"""
        return {
            'current_metrics': self.alert_metrics.copy(),
            'active_alerts_count': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'enabled_rules': len([r for r in self.alert_rules.values() if r.enabled]),
            'notification_channels': len(self.escalation_manager.notification_channels),
            'queue_size': self.alert_queue.qsize() if hasattr(self.alert_queue, 'qsize') else 0
        }
    
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule"""
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"📋 Added alert rule: {rule.name}")
    
    def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing alert rule"""
        rule = self.alert_rules.get(rule_id)
        if not rule:
            return False
        
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        logger.info(f"📋 Updated alert rule: {rule_id}")
        return True
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add notification channel"""
        self.escalation_manager.add_notification_channel(channel)
        logger.info(f"📢 Added notification channel: {channel.name}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        return {
            'running': self.is_running,
            'active_alerts': len(self.active_alerts),
            'total_rules': len(self.alert_rules),
            'processing_queue_size': self.alert_queue.qsize() if hasattr(self.alert_queue, 'qsize') else 0,
            'notification_channels': len(self.escalation_manager.notification_channels),
            'last_metrics_update': datetime.now().isoformat()
        }