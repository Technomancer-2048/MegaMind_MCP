#!/usr/bin/env python3
"""
Realm Health Monitoring and Alerting System for MegaMind Context Database
Provides real-time monitoring, alerting, and automated health checks
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Callable
from enum import Enum

try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"

class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ComponentType(Enum):
    DATABASE = "database"
    INHERITANCE = "inheritance"
    SEARCH = "search"
    PROMOTION = "promotion"
    SECURITY = "security"
    REALM = "realm"

@dataclass
class HealthCheck:
    """Represents a health check result"""
    check_id: str
    component_type: ComponentType
    component_name: str
    status: HealthStatus
    health_score: float
    performance_metrics: Dict[str, Any]
    check_timestamp: datetime
    next_check: datetime
    details: Optional[str] = None

@dataclass
class Alert:
    """Represents a monitoring alert"""
    alert_id: str
    realm_id: Optional[str]
    component_type: ComponentType
    component_name: str
    severity: AlertSeverity
    title: str
    description: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolution_notes: Optional[str] = None

@dataclass
class MonitoringRule:
    """Represents a monitoring rule configuration"""
    rule_id: str
    component_type: ComponentType
    metric_name: str
    operator: str  # 'gt', 'lt', 'eq', 'ne'
    threshold_value: float
    severity: AlertSeverity
    check_interval_minutes: int
    enabled: bool = True

class RealmMonitoring:
    """Comprehensive monitoring and alerting system for realms"""
    
    def __init__(self, db_connection, config: Dict[str, Any] = None):
        self.db = db_connection
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.alert_handlers = []
        self._setup_default_rules()
    
    # ===================================================================
    # Health Check System
    # ===================================================================
    
    def perform_health_check(self, component_type: ComponentType, 
                           component_name: str) -> HealthCheck:
        """Perform health check for a specific component"""
        try:
            check_id = f"health_{uuid.uuid4().hex[:12]}"
            
            if component_type == ComponentType.DATABASE:
                return self._check_database_health(check_id, component_name)
            elif component_type == ComponentType.REALM:
                return self._check_realm_health(check_id, component_name)
            elif component_type == ComponentType.SEARCH:
                return self._check_search_health(check_id, component_name)
            elif component_type == ComponentType.INHERITANCE:
                return self._check_inheritance_health(check_id, component_name)
            elif component_type == ComponentType.PROMOTION:
                return self._check_promotion_health(check_id, component_name)
            elif component_type == ComponentType.SECURITY:
                return self._check_security_health(check_id, component_name)
            else:
                return self._create_default_health_check(check_id, component_type, component_name)
                
        except Exception as e:
            self.logger.error(f"Health check failed for {component_type.value}/{component_name}: {e}")
            return self._create_failed_health_check(
                f"health_{uuid.uuid4().hex[:12]}", component_type, component_name, str(e)
            )
    
    def _check_database_health(self, check_id: str, component_name: str) -> HealthCheck:
        """Check database component health"""
        try:
            cursor = self.db.cursor(dictionary=True)
            start_time = datetime.now()
            
            # Test basic connectivity and performance
            cursor.execute("SELECT 1 as test")
            cursor.fetchone()
            
            # Get connection statistics
            cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
            connections_result = cursor.fetchone()
            active_connections = int(connections_result['Value']) if connections_result else 0
            
            # Get slow query count
            cursor.execute("SHOW STATUS LIKE 'Slow_queries'")
            slow_queries_result = cursor.fetchone()
            slow_queries = int(slow_queries_result['Value']) if slow_queries_result else 0
            
            # Calculate response time
            response_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine health status
            health_score = 100.0
            status = HealthStatus.HEALTHY
            
            if response_time_ms > 1000:  # > 1 second is concerning
                health_score -= 30
                status = HealthStatus.WARNING
            elif response_time_ms > 5000:  # > 5 seconds is critical
                health_score -= 60
                status = HealthStatus.CRITICAL
            
            if active_connections > 80:  # Assuming max 100 connections
                health_score -= 20
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
            
            performance_metrics = {
                "response_time_ms": response_time_ms,
                "active_connections": active_connections,
                "slow_queries": slow_queries,
                "max_connections": 100  # Would be configurable
            }
            
            return HealthCheck(
                check_id=check_id,
                component_type=ComponentType.DATABASE,
                component_name=component_name,
                status=status,
                health_score=max(health_score, 0),
                performance_metrics=performance_metrics,
                check_timestamp=datetime.now(),
                next_check=datetime.now() + timedelta(minutes=5)
            )
            
        except Exception as e:
            return self._create_failed_health_check(
                check_id, ComponentType.DATABASE, component_name, str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    def _check_realm_health(self, check_id: str, realm_id: str) -> HealthCheck:
        """Check realm-specific health"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check realm exists and is active
            cursor.execute("SELECT * FROM megamind_realms WHERE realm_id = %s", (realm_id,))
            realm = cursor.fetchone()
            
            if not realm or not realm['is_active']:
                return self._create_failed_health_check(
                    check_id, ComponentType.REALM, realm_id, "Realm not found or inactive"
                )
            
            # Get realm statistics
            cursor.execute("""
                SELECT 
                    COUNT(c.chunk_id) as chunk_count,
                    AVG(c.access_count) as avg_access,
                    COUNT(CASE WHEN c.last_accessed >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as recent_access_count,
                    COUNT(cr.relationship_id) as relationship_count
                FROM megamind_chunks c
                LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id
                WHERE c.realm_id = %s
            """, (realm_id,))
            
            stats = cursor.fetchone()
            
            # Calculate health score
            health_score = 100.0
            status = HealthStatus.HEALTHY
            
            chunk_count = stats['chunk_count'] or 0
            avg_access = stats['avg_access'] or 0
            recent_access_count = stats['recent_access_count'] or 0
            relationship_count = stats['relationship_count'] or 0
            
            # Penalize empty or underutilized realms
            if chunk_count == 0:
                health_score = 0
                status = HealthStatus.CRITICAL
            elif chunk_count < 5:
                health_score -= 30
                status = HealthStatus.WARNING
            
            if avg_access < 1.5:
                health_score -= 20
            
            if recent_access_count == 0 and chunk_count > 0:
                health_score -= 25
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
            
            # Relationship density bonus
            if chunk_count > 0:
                relationship_density = relationship_count / chunk_count
                if relationship_density > 0.3:
                    health_score += 10  # Bonus for good relationships
            
            performance_metrics = {
                "chunk_count": chunk_count,
                "avg_access_count": float(avg_access),
                "recent_access_count": recent_access_count,
                "relationship_count": relationship_count,
                "relationship_density": relationship_count / max(chunk_count, 1)
            }
            
            return HealthCheck(
                check_id=check_id,
                component_type=ComponentType.REALM,
                component_name=realm_id,
                status=status,
                health_score=max(min(health_score, 100), 0),
                performance_metrics=performance_metrics,
                check_timestamp=datetime.now(),
                next_check=datetime.now() + timedelta(minutes=15)
            )
            
        except Exception as e:
            return self._create_failed_health_check(
                check_id, ComponentType.REALM, realm_id, str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    def _check_search_health(self, check_id: str, component_name: str) -> HealthCheck:
        """Check search system health"""
        try:
            cursor = self.db.cursor()
            start_time = datetime.now()
            
            # Perform test search
            test_queries = ["security", "database", "api"]
            total_results = 0
            
            for query in test_queries:
                search_query = """
                SELECT COUNT(*) as result_count
                FROM megamind_chunks c
                WHERE c.content LIKE %s
                LIMIT 10
                """
                cursor.execute(search_query, (f"%{query}%",))
                result = cursor.fetchone()
                total_results += result[0] if result else 0
            
            search_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            # Health assessment
            health_score = 100.0
            status = HealthStatus.HEALTHY
            
            if search_time_ms > 2000:  # > 2 seconds is slow
                health_score -= 40
                status = HealthStatus.WARNING
            elif search_time_ms > 5000:  # > 5 seconds is critical
                health_score -= 70
                status = HealthStatus.CRITICAL
            
            if total_results == 0:
                health_score -= 50
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
            
            performance_metrics = {
                "search_time_ms": search_time_ms,
                "test_results_found": total_results,
                "test_queries_count": len(test_queries)
            }
            
            return HealthCheck(
                check_id=check_id,
                component_type=ComponentType.SEARCH,
                component_name=component_name,
                status=status,
                health_score=max(health_score, 0),
                performance_metrics=performance_metrics,
                check_timestamp=datetime.now(),
                next_check=datetime.now() + timedelta(minutes=10)
            )
            
        except Exception as e:
            return self._create_failed_health_check(
                check_id, ComponentType.SEARCH, component_name, str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    def _check_inheritance_health(self, check_id: str, component_name: str) -> HealthCheck:
        """Check inheritance system health"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check inheritance relationships
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_relationships,
                    COUNT(DISTINCT child_realm_id) as realms_with_inheritance,
                    COUNT(DISTINCT parent_realm_id) as parent_realms
                FROM megamind_realm_inheritance
            """)
            
            inheritance_stats = cursor.fetchone()
            
            # Test inheritance resolution (simplified)
            cursor.execute("""
                SELECT COUNT(DISTINCT c.chunk_id) as accessible_chunks
                FROM megamind_chunks c
                JOIN megamind_realms r ON c.realm_id = r.realm_id
                WHERE r.is_active = TRUE
            """)
            
            accessibility_stats = cursor.fetchone()
            
            health_score = 100.0
            status = HealthStatus.HEALTHY
            
            total_relationships = inheritance_stats['total_relationships'] or 0
            realms_with_inheritance = inheritance_stats['realms_with_inheritance'] or 0
            
            if total_relationships == 0:
                health_score -= 30  # No inheritance setup is concerning
                status = HealthStatus.WARNING
            
            performance_metrics = {
                "total_inheritance_relationships": total_relationships,
                "realms_with_inheritance": realms_with_inheritance,
                "parent_realms": inheritance_stats['parent_realms'] or 0,
                "accessible_chunks": accessibility_stats['accessible_chunks'] or 0
            }
            
            return HealthCheck(
                check_id=check_id,
                component_type=ComponentType.INHERITANCE,
                component_name=component_name,
                status=status,
                health_score=health_score,
                performance_metrics=performance_metrics,
                check_timestamp=datetime.now(),
                next_check=datetime.now() + timedelta(minutes=30)
            )
            
        except Exception as e:
            return self._create_failed_health_check(
                check_id, ComponentType.INHERITANCE, component_name, str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    def _check_promotion_health(self, check_id: str, component_name: str) -> HealthCheck:
        """Check promotion system health"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check promotion queue status
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_requests,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_requests,
                    COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved_requests,
                    COUNT(CASE WHEN status = 'rejected' THEN 1 END) as rejected_requests,
                    AVG(TIMESTAMPDIFF(HOUR, requested_at, COALESCE(reviewed_at, NOW()))) as avg_processing_hours
                FROM megamind_promotion_queue
                WHERE requested_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            """)
            
            promotion_stats = cursor.fetchone()
            
            health_score = 100.0
            status = HealthStatus.HEALTHY
            
            pending_requests = promotion_stats['pending_requests'] or 0
            total_requests = promotion_stats['total_requests'] or 0
            avg_processing_hours = promotion_stats['avg_processing_hours'] or 0
            
            # Check for bottlenecks
            if pending_requests > 10:
                health_score -= 30
                status = HealthStatus.WARNING
            elif pending_requests > 25:
                health_score -= 60
                status = HealthStatus.CRITICAL
            
            if avg_processing_hours > 72:  # More than 3 days
                health_score -= 25
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
            
            performance_metrics = {
                "total_requests_30_days": total_requests,
                "pending_requests": pending_requests,
                "approved_requests": promotion_stats['approved_requests'] or 0,
                "rejected_requests": promotion_stats['rejected_requests'] or 0,
                "avg_processing_hours": float(avg_processing_hours)
            }
            
            return HealthCheck(
                check_id=check_id,
                component_type=ComponentType.PROMOTION,
                component_name=component_name,
                status=status,
                health_score=health_score,
                performance_metrics=performance_metrics,
                check_timestamp=datetime.now(),
                next_check=datetime.now() + timedelta(minutes=20)
            )
            
        except Exception as e:
            return self._create_failed_health_check(
                check_id, ComponentType.PROMOTION, component_name, str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    def _check_security_health(self, check_id: str, component_name: str) -> HealthCheck:
        """Check security system health"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check recent security violations
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_violations,
                    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_violations,
                    COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_violations,
                    COUNT(CASE WHEN detected_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR) THEN 1 END) as recent_violations
                FROM megamind_security_violations
                WHERE detected_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            """)
            
            security_stats = cursor.fetchone()
            
            # Check failed access attempts from audit log
            cursor.execute("""
                SELECT COUNT(*) as failed_attempts
                FROM megamind_audit_log
                WHERE event_type LIKE '%failed%'
                AND event_timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            """)
            
            access_stats = cursor.fetchone()
            
            health_score = 100.0
            status = HealthStatus.HEALTHY
            
            critical_violations = security_stats['critical_violations'] or 0
            high_violations = security_stats['high_violations'] or 0
            recent_violations = security_stats['recent_violations'] or 0
            failed_attempts = access_stats['failed_attempts'] or 0
            
            # Security scoring
            if critical_violations > 0:
                health_score -= 50
                status = HealthStatus.CRITICAL
            elif high_violations > 2:
                health_score -= 30
                status = HealthStatus.WARNING
            
            if recent_violations > 5:
                health_score -= 25
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
            
            if failed_attempts > 20:  # More than 20 failed attempts per hour
                health_score -= 20
                status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
            
            performance_metrics = {
                "violations_24h": security_stats['total_violations'] or 0,
                "critical_violations_24h": critical_violations,
                "high_violations_24h": high_violations,
                "recent_violations_1h": recent_violations,
                "failed_attempts_1h": failed_attempts
            }
            
            return HealthCheck(
                check_id=check_id,
                component_type=ComponentType.SECURITY,
                component_name=component_name,
                status=status,
                health_score=health_score,
                performance_metrics=performance_metrics,
                check_timestamp=datetime.now(),
                next_check=datetime.now() + timedelta(minutes=5)
            )
            
        except Exception as e:
            return self._create_failed_health_check(
                check_id, ComponentType.SECURITY, component_name, str(e)
            )
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Alert Management
    # ===================================================================
    
    def check_monitoring_rules(self, health_check: HealthCheck) -> List[Alert]:
        """Check health check against monitoring rules and generate alerts"""
        alerts = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Get monitoring rules for this component type
            cursor.execute("""
                SELECT * FROM megamind_monitoring_rules 
                WHERE component_type = %s AND enabled = TRUE
            """, (health_check.component_type.value,))
            
            rules = cursor.fetchall()
            
            for rule in rules:
                alert = self._evaluate_rule(rule, health_check)
                if alert:
                    alerts.append(alert)
            
            # Store alerts in database
            for alert in alerts:
                self._store_alert(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to check monitoring rules: {e}")
            return alerts
        finally:
            if cursor:
                cursor.close()
    
    def _evaluate_rule(self, rule: Dict, health_check: HealthCheck) -> Optional[Alert]:
        """Evaluate a monitoring rule against health check results"""
        try:
            metric_name = rule['metric_name']
            operator = rule['operator']
            threshold = rule['threshold_value']
            
            # Get metric value from health check
            if metric_name == 'health_score':
                metric_value = health_check.health_score
            elif metric_name in health_check.performance_metrics:
                metric_value = health_check.performance_metrics[metric_name]
            else:
                return None
            
            # Evaluate condition
            triggered = False
            if operator == 'gt' and metric_value > threshold:
                triggered = True
            elif operator == 'lt' and metric_value < threshold:
                triggered = True
            elif operator == 'eq' and metric_value == threshold:
                triggered = True
            elif operator == 'ne' and metric_value != threshold:
                triggered = True
            
            if triggered:
                return Alert(
                    alert_id=f"alert_{uuid.uuid4().hex[:12]}",
                    realm_id=health_check.component_name if health_check.component_type == ComponentType.REALM else None,
                    component_type=health_check.component_type,
                    component_name=health_check.component_name,
                    severity=AlertSeverity(rule['severity']),
                    title=f"{health_check.component_type.value.title()} {metric_name} Alert",
                    description=f"{metric_name} is {metric_value}, threshold: {operator} {threshold}",
                    triggered_at=datetime.now()
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate rule: {e}")
            return None
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            cursor = self.db.cursor()
            
            query = """
            INSERT INTO megamind_monitoring_alerts 
            (alert_id, realm_id, component_type, component_name, severity, title, description, triggered_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (
                alert.alert_id, alert.realm_id, alert.component_type.value,
                alert.component_name, alert.severity.value, alert.title,
                alert.description, alert.triggered_at
            ))
            
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")
            self.db.rollback()
        finally:
            if cursor:
                cursor.close()
    
    def send_alert_notifications(self, alerts: List[Alert]):
        """Send notifications for alerts"""
        for alert in alerts:
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
    
    # ===================================================================
    # Alert Handlers
    # ===================================================================
    
    def add_email_alert_handler(self, smtp_config: Dict[str, str], recipients: List[str]):
        """Add email alert handler"""
        if not EMAIL_AVAILABLE:
            self.logger.warning("Email libraries not available, email alerts disabled")
            return
            
        def email_handler(alert: Alert):
            try:
                subject = f"[MegaMind] {alert.severity.value.upper()}: {alert.title}"
                
                body = f"""
                Alert Details:
                - Component: {alert.component_type.value}/{alert.component_name}
                - Severity: {alert.severity.value}
                - Time: {alert.triggered_at}
                - Description: {alert.description}
                
                Please investigate and take appropriate action.
                """
                
                msg = MimeMultipart()
                msg['From'] = smtp_config['from_email']
                msg['To'] = ', '.join(recipients)
                msg['Subject'] = subject
                msg.attach(MimeText(body, 'plain'))
                
                server = smtplib.SMTP(smtp_config['smtp_host'], smtp_config['smtp_port'])
                if smtp_config.get('use_tls'):
                    server.starttls()
                if smtp_config.get('username'):
                    server.login(smtp_config['username'], smtp_config['password'])
                
                server.send_message(msg)
                server.quit()
                
                self.logger.info(f"Email alert sent for {alert.alert_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to send email alert: {e}")
        
        self.alert_handlers.append(email_handler)
    
    def add_webhook_alert_handler(self, webhook_url: str):
        """Add webhook alert handler"""
        def webhook_handler(alert: Alert):
            try:
                import requests
                
                payload = {
                    "alert_id": alert.alert_id,
                    "component": f"{alert.component_type.value}/{alert.component_name}",
                    "severity": alert.severity.value,
                    "title": alert.title,
                    "description": alert.description,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "realm_id": alert.realm_id
                }
                
                response = requests.post(webhook_url, json=payload, timeout=10)
                response.raise_for_status()
                
                self.logger.info(f"Webhook alert sent for {alert.alert_id}")
                
            except Exception as e:
                self.logger.error(f"Failed to send webhook alert: {e}")
        
        self.alert_handlers.append(webhook_handler)
    
    # ===================================================================
    # Monitoring Orchestration
    # ===================================================================
    
    def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        try:
            self.logger.info("Starting monitoring cycle")
            
            # Define components to monitor
            components = [
                (ComponentType.DATABASE, "connection_pool"),
                (ComponentType.DATABASE, "query_performance"),
                (ComponentType.SEARCH, "chunk_search"),
                (ComponentType.INHERITANCE, "resolution_engine"),
                (ComponentType.PROMOTION, "workflow_engine"),
                (ComponentType.SECURITY, "access_control")
            ]
            
            all_alerts = []
            
            # Monitor system components
            for component_type, component_name in components:
                health_check = self.perform_health_check(component_type, component_name)
                self._update_system_health(health_check)
                
                alerts = self.check_monitoring_rules(health_check)
                all_alerts.extend(alerts)
            
            # Monitor individual realms
            realm_alerts = self._monitor_all_realms()
            all_alerts.extend(realm_alerts)
            
            # Send notifications for new alerts
            if all_alerts:
                self.send_alert_notifications(all_alerts)
                self.logger.info(f"Monitoring cycle completed with {len(all_alerts)} alerts")
            else:
                self.logger.info("Monitoring cycle completed - all systems healthy")
            
        except Exception as e:
            self.logger.error(f"Monitoring cycle failed: {e}")
    
    def _monitor_all_realms(self) -> List[Alert]:
        """Monitor all active realms"""
        alerts = []
        
        try:
            cursor = self.db.cursor()
            cursor.execute("SELECT realm_id FROM megamind_realms WHERE is_active = TRUE")
            realms = cursor.fetchall()
            
            for (realm_id,) in realms:
                health_check = self.perform_health_check(ComponentType.REALM, realm_id)
                self._update_system_health(health_check)
                
                realm_alerts = self.check_monitoring_rules(health_check)
                alerts.extend(realm_alerts)
            
        except Exception as e:
            self.logger.error(f"Failed to monitor realms: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return alerts
    
    def _update_system_health(self, health_check: HealthCheck):
        """Update system health table with check results"""
        try:
            cursor = self.db.cursor()
            
            query = """
            INSERT INTO megamind_system_health 
            (health_id, component_type, component_name, status, health_score, performance_metrics, last_check)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            status = VALUES(status),
            health_score = VALUES(health_score),
            performance_metrics = VALUES(performance_metrics),
            last_check = VALUES(last_check)
            """
            
            cursor.execute(query, (
                health_check.check_id,
                health_check.component_type.value,
                health_check.component_name,
                health_check.status.value,
                health_check.health_score,
                json.dumps(health_check.performance_metrics),
                health_check.check_timestamp
            ))
            
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to update system health: {e}")
            self.db.rollback()
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Helper Methods
    # ===================================================================
    
    def _setup_default_rules(self):
        """Set up default monitoring rules"""
        # This would typically load from configuration or database
        pass
    
    def _create_default_health_check(self, check_id: str, component_type: ComponentType, 
                                   component_name: str) -> HealthCheck:
        """Create a default health check for unknown components"""
        return HealthCheck(
            check_id=check_id,
            component_type=component_type,
            component_name=component_name,
            status=HealthStatus.HEALTHY,
            health_score=100.0,
            performance_metrics={},
            check_timestamp=datetime.now(),
            next_check=datetime.now() + timedelta(minutes=15)
        )
    
    def _create_failed_health_check(self, check_id: str, component_type: ComponentType,
                                  component_name: str, error_message: str) -> HealthCheck:
        """Create a failed health check"""
        return HealthCheck(
            check_id=check_id,
            component_type=component_type,
            component_name=component_name,
            status=HealthStatus.CRITICAL,
            health_score=0.0,
            performance_metrics={"error": error_message},
            check_timestamp=datetime.now(),
            next_check=datetime.now() + timedelta(minutes=5),
            details=error_message
        )