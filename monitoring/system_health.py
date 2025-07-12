#!/usr/bin/env python3
"""
System Health Monitoring for MegaMind Context Database
Phase 4: Advanced Optimization

Comprehensive monitoring system with performance metrics, alerting,
and system health assessment capabilities.
"""

import json
import logging
import os
import time
import psutil
import threading
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque, defaultdict

import mysql.connector
from mysql.connector import pooling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class MetricSample:
    """Individual metric sample"""
    timestamp: datetime
    metric_name: str
    value: float
    tags: Dict[str, str]

@dataclass
class HealthCheck:
    """Health check result"""
    check_name: str
    status: str  # healthy, warning, critical
    message: str
    timestamp: datetime
    metrics: Dict[str, float]

@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    metric_name: str
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    duration_minutes: int
    severity: str  # info, warning, critical
    enabled: bool

@dataclass
class Alert:
    """Alert instance"""
    alert_id: str
    rule_id: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str
    message: str
    timestamp: datetime
    resolved: bool

class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection_pool = None
        self.metrics_buffer = deque(maxlen=10000)  # Store last 10k metrics
        self.collection_interval = 30  # seconds
        self.running = False
        self.collection_thread = None
        self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """Setup MySQL connection pool"""
        try:
            pool_config = {
                'pool_name': 'monitoring_pool',
                'pool_size': 3,
                'host': self.db_config['host'],
                'port': int(self.db_config['port']),
                'database': self.db_config['database'],
                'user': self.db_config['user'],
                'password': self.db_config['password'],
                'autocommit': False,
                'charset': 'utf8mb4',
                'use_unicode': True
            }
            self.connection_pool = pooling.MySQLConnectionPool(**pool_config)
            logger.info("Monitoring system database connection pool established")
        except Exception as e:
            logger.error(f"Failed to setup monitoring database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get a connection from the pool"""
        return self.connection_pool.get_connection()
    
    def start_collection(self):
        """Start metrics collection in background thread"""
        if not self.running:
            self.running = True
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
            logger.info("Metrics collection started")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        logger.info("Metrics collection stopped")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                # Collect all metrics
                metrics = []
                metrics.extend(self._collect_database_metrics())
                metrics.extend(self._collect_system_metrics())
                metrics.extend(self._collect_application_metrics())
                
                # Store metrics
                for metric in metrics:
                    self.metrics_buffer.append(metric)
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_database_metrics(self) -> List[MetricSample]:
        """Collect database performance metrics"""
        metrics = []
        timestamp = datetime.now()
        
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Connection pool metrics
            try:
                pool_size = self.connection_pool.pool_size
                # This is an approximation - MySQL Connector doesn't expose active connections directly
                metrics.append(MetricSample(
                    timestamp=timestamp,
                    metric_name="db_connection_pool_size",
                    value=float(pool_size),
                    tags={"database": self.db_config['database']}
                ))
            except:
                pass
            
            # Database size metrics
            cursor.execute("""
                SELECT table_name, 
                       ROUND(((data_length + index_length) / 1024 / 1024), 2) AS size_mb
                FROM information_schema.tables 
                WHERE table_schema = %s AND table_name LIKE 'megamind_%'
            """, (self.db_config['database'],))
            
            total_size = 0
            for row in cursor.fetchall():
                table_size = float(row['size_mb'])
                total_size += table_size
                metrics.append(MetricSample(
                    timestamp=timestamp,
                    metric_name="db_table_size_mb",
                    value=table_size,
                    tags={"table": row['table_name']}
                ))
            
            metrics.append(MetricSample(
                timestamp=timestamp,
                metric_name="db_total_size_mb",
                value=total_size,
                tags={"database": self.db_config['database']}
            ))
            
            # Chunk metrics
            cursor.execute("SELECT COUNT(*) as total_chunks FROM megamind_chunks")
            total_chunks = cursor.fetchone()['total_chunks']
            
            cursor.execute("SELECT COUNT(*) as hot_chunks FROM megamind_chunks WHERE access_count > 10")
            hot_chunks = cursor.fetchone()['hot_chunks']
            
            cursor.execute("SELECT COUNT(*) as cold_chunks FROM megamind_chunks WHERE access_count = 0")
            cold_chunks = cursor.fetchone()['cold_chunks']
            
            cursor.execute("SELECT AVG(access_count) as avg_access FROM megamind_chunks")
            avg_access = float(cursor.fetchone()['avg_access'] or 0)
            
            metrics.extend([
                MetricSample(timestamp, "chunks_total", float(total_chunks), {}),
                MetricSample(timestamp, "chunks_hot", float(hot_chunks), {}),
                MetricSample(timestamp, "chunks_cold", float(cold_chunks), {}),
                MetricSample(timestamp, "chunks_avg_access", avg_access, {}),
            ])
            
            # Relationship metrics
            cursor.execute("SELECT COUNT(*) as total_relationships FROM megamind_chunk_relationships")
            total_relationships = cursor.fetchone()['total_relationships']
            
            cursor.execute("""
                SELECT AVG(relationship_count) as avg_relationships
                FROM (
                    SELECT chunk_id, COUNT(*) as relationship_count 
                    FROM megamind_chunk_relationships 
                    GROUP BY chunk_id
                ) t
            """)
            avg_relationships = float(cursor.fetchone()['avg_relationships'] or 0)
            
            metrics.extend([
                MetricSample(timestamp, "relationships_total", float(total_relationships), {}),
                MetricSample(timestamp, "relationships_avg_per_chunk", avg_relationships, {}),
            ])
            
            # Session and change metrics
            cursor.execute("SELECT COUNT(*) as pending_sessions FROM megamind_session_metadata WHERE pending_changes_count > 0")
            pending_sessions = cursor.fetchone()['pending_sessions']
            
            cursor.execute("SELECT COUNT(*) as total_changes FROM megamind_session_changes WHERE status = 'pending'")
            total_changes = cursor.fetchone()['total_changes']
            
            metrics.extend([
                MetricSample(timestamp, "sessions_with_pending_changes", float(pending_sessions), {}),
                MetricSample(timestamp, "total_pending_changes", float(total_changes), {}),
            ])
            
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {e}")
        finally:
            if connection:
                connection.close()
        
        return metrics
    
    def _collect_system_metrics(self) -> List[MetricSample]:
        """Collect system-level metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(MetricSample(timestamp, "system_cpu_percent", cpu_percent, {}))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.extend([
                MetricSample(timestamp, "system_memory_total_gb", memory.total / (1024**3), {}),
                MetricSample(timestamp, "system_memory_used_gb", memory.used / (1024**3), {}),
                MetricSample(timestamp, "system_memory_percent", memory.percent, {}),
            ])
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.extend([
                MetricSample(timestamp, "system_disk_total_gb", disk.total / (1024**3), {}),
                MetricSample(timestamp, "system_disk_used_gb", disk.used / (1024**3), {}),
                MetricSample(timestamp, "system_disk_percent", (disk.used / disk.total) * 100, {}),
            ])
            
            # Network metrics (if available)
            try:
                network = psutil.net_io_counters()
                metrics.extend([
                    MetricSample(timestamp, "network_bytes_sent", float(network.bytes_sent), {}),
                    MetricSample(timestamp, "network_bytes_recv", float(network.bytes_recv), {}),
                ])
            except:
                pass
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def _collect_application_metrics(self) -> List[MetricSample]:
        """Collect application-specific metrics"""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Process metrics for current Python process
            current_process = psutil.Process()
            
            # Memory usage of current process
            memory_info = current_process.memory_info()
            metrics.extend([
                MetricSample(timestamp, "app_memory_rss_mb", memory_info.rss / (1024**2), {}),
                MetricSample(timestamp, "app_memory_vms_mb", memory_info.vms / (1024**2), {}),
            ])
            
            # CPU usage of current process
            cpu_percent = current_process.cpu_percent()
            metrics.append(MetricSample(timestamp, "app_cpu_percent", cpu_percent, {}))
            
            # File descriptor count
            try:
                num_fds = current_process.num_fds()
                metrics.append(MetricSample(timestamp, "app_file_descriptors", float(num_fds), {}))
            except:
                pass
            
            # Thread count
            num_threads = current_process.num_threads()
            metrics.append(MetricSample(timestamp, "app_threads", float(num_threads), {}))
            
        except Exception as e:
            logger.error(f"Failed to collect application metrics: {e}")
        
        return metrics
    
    def get_recent_metrics(self, metric_name: str, minutes: int = 60) -> List[MetricSample]:
        """Get recent metrics for a specific metric name"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metric for metric in self.metrics_buffer 
            if metric.metric_name == metric_name and metric.timestamp >= cutoff_time
        ]
    
    def get_metric_summary(self, metric_name: str, minutes: int = 60) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        recent_metrics = self.get_recent_metrics(metric_name, minutes)
        
        if not recent_metrics:
            return {}
        
        values = [metric.value for metric in recent_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0
        }

class AlertManager:
    """Manages alerting rules and notifications"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.evaluation_interval = 60  # seconds
        self.running = False
        self.evaluation_thread = None
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alerting rules"""
        default_rules = [
            AlertRule("cpu_high", "system_cpu_percent", "gt", 80.0, 5, "warning", True),
            AlertRule("cpu_critical", "system_cpu_percent", "gt", 95.0, 2, "critical", True),
            AlertRule("memory_high", "system_memory_percent", "gt", 85.0, 5, "warning", True),
            AlertRule("memory_critical", "system_memory_percent", "gt", 95.0, 2, "critical", True),
            AlertRule("disk_high", "system_disk_percent", "gt", 80.0, 10, "warning", True),
            AlertRule("disk_critical", "system_disk_percent", "gt", 90.0, 5, "critical", True),
            AlertRule("db_size_large", "db_total_size_mb", "gt", 1000.0, 30, "info", True),
            AlertRule("cold_chunks_high", "chunks_cold", "gt", 1000.0, 60, "warning", True),
            AlertRule("pending_changes_high", "total_pending_changes", "gt", 100.0, 30, "warning", True),
        ]
        
        self.alert_rules.extend(default_rules)
        logger.info(f"Loaded {len(default_rules)} default alert rules")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule"""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.rule_id}")
    
    def start_monitoring(self):
        """Start alert monitoring"""
        if not self.running:
            self.running = True
            self.evaluation_thread = threading.Thread(target=self._evaluation_loop, daemon=True)
            self.evaluation_thread.start()
            logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.running = False
        if self.evaluation_thread:
            self.evaluation_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
    
    def _evaluation_loop(self):
        """Main alert evaluation loop"""
        while self.running:
            try:
                self._evaluate_all_rules()
                time.sleep(self.evaluation_interval)
            except Exception as e:
                logger.error(f"Error in alert evaluation: {e}")
                time.sleep(self.evaluation_interval)
    
    def _evaluate_all_rules(self):
        """Evaluate all alert rules"""
        for rule in self.alert_rules:
            if rule.enabled:
                self._evaluate_rule(rule)
    
    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        try:
            # Get recent metrics for this rule
            recent_metrics = self.metrics_collector.get_recent_metrics(
                rule.metric_name, 
                rule.duration_minutes
            )
            
            if not recent_metrics:
                return
            
            # Check if condition is met for the required duration
            current_time = datetime.now()
            duration_cutoff = current_time - timedelta(minutes=rule.duration_minutes)
            
            # Get metrics within the duration window
            relevant_metrics = [
                metric for metric in recent_metrics 
                if metric.timestamp >= duration_cutoff
            ]
            
            if not relevant_metrics:
                return
            
            # Check if all recent metrics meet the condition
            condition_met = all(
                self._check_condition(metric.value, rule.condition, rule.threshold)
                for metric in relevant_metrics
            )
            
            current_value = relevant_metrics[-1].value if relevant_metrics else 0
            
            if condition_met:
                # Condition is met - trigger alert if not already active
                if rule.rule_id not in self.active_alerts:
                    alert = Alert(
                        alert_id=f"{rule.rule_id}_{int(current_time.timestamp())}",
                        rule_id=rule.rule_id,
                        metric_name=rule.metric_name,
                        current_value=current_value,
                        threshold=rule.threshold,
                        severity=rule.severity,
                        message=f"{rule.metric_name} is {current_value:.2f} (threshold: {rule.threshold})",
                        timestamp=current_time,
                        resolved=False
                    )
                    
                    self.active_alerts[rule.rule_id] = alert
                    self.alert_history.append(alert)
                    self._send_alert(alert)
            else:
                # Condition not met - resolve alert if active
                if rule.rule_id in self.active_alerts:
                    alert = self.active_alerts[rule.rule_id]
                    alert.resolved = True
                    self._send_resolution(alert)
                    del self.active_alerts[rule.rule_id]
        
        except Exception as e:
            logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if a value meets the condition"""
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001  # Float equality
        else:
            return False
    
    def _send_alert(self, alert: Alert):
        """Send alert notification"""
        logger.warning(f"ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Here you could integrate with external alerting systems:
        # - Send email
        # - Post to Slack
        # - Send to PagerDuty
        # - Write to alert log file
        
        # For now, just log the alert
        alert_data = {
            "alert_id": alert.alert_id,
            "rule_id": alert.rule_id,
            "metric_name": alert.metric_name,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "severity": alert.severity,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat()
        }
        
        # Write to alert log
        self._write_alert_log(alert_data, "triggered")
    
    def _send_resolution(self, alert: Alert):
        """Send alert resolution notification"""
        logger.info(f"RESOLVED: Alert {alert.alert_id} for {alert.metric_name}")
        
        alert_data = {
            "alert_id": alert.alert_id,
            "rule_id": alert.rule_id,
            "metric_name": alert.metric_name,
            "resolved_at": datetime.now().isoformat()
        }
        
        self._write_alert_log(alert_data, "resolved")
    
    def _write_alert_log(self, alert_data: Dict[str, Any], action: str):
        """Write alert to log file"""
        try:
            log_dir = Path("monitoring_logs")
            log_dir.mkdir(exist_ok=True)
            
            log_file = log_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.json"
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "alert_data": alert_data
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write alert log: {e}")

class SystemHealthMonitor:
    """Main system health monitoring coordinator"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.metrics_collector = MetricsCollector(db_config)
        self.alert_manager = AlertManager(self.metrics_collector)
        self.health_checks = []
        self._setup_health_checks()
    
    def _setup_health_checks(self):
        """Setup health check functions"""
        self.health_checks = [
            ("database_connectivity", self._check_database_connectivity),
            ("system_resources", self._check_system_resources),
            ("chunk_distribution", self._check_chunk_distribution),
            ("relationship_integrity", self._check_relationship_integrity),
            ("performance_metrics", self._check_performance_metrics),
        ]
    
    def start(self):
        """Start monitoring system"""
        logger.info("Starting MegaMind system health monitoring...")
        self.metrics_collector.start_collection()
        self.alert_manager.start_monitoring()
        logger.info("System health monitoring started")
    
    def stop(self):
        """Stop monitoring system"""
        logger.info("Stopping system health monitoring...")
        self.alert_manager.stop_monitoring()
        self.metrics_collector.stop_collection()
        logger.info("System health monitoring stopped")
    
    def run_health_checks(self) -> List[HealthCheck]:
        """Run all health checks"""
        results = []
        
        for check_name, check_function in self.health_checks:
            try:
                result = check_function()
                results.append(result)
            except Exception as e:
                results.append(HealthCheck(
                    check_name=check_name,
                    status="critical",
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now(),
                    metrics={}
                ))
        
        return results
    
    def _check_database_connectivity(self) -> HealthCheck:
        """Check database connectivity and basic queries"""
        connection = None
        try:
            connection = self.metrics_collector.get_connection()
            cursor = connection.cursor()
            
            # Test basic connectivity
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
            # Test MegaMind tables
            cursor.execute("SELECT COUNT(*) FROM megamind_chunks")
            chunk_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM megamind_chunk_relationships")
            relationship_count = cursor.fetchone()[0]
            
            return HealthCheck(
                check_name="database_connectivity",
                status="healthy",
                message="Database connectivity and basic queries successful",
                timestamp=datetime.now(),
                metrics={
                    "chunk_count": float(chunk_count),
                    "relationship_count": float(relationship_count)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="database_connectivity",
                status="critical",
                message=f"Database connectivity failed: {str(e)}",
                timestamp=datetime.now(),
                metrics={}
            )
        finally:
            if connection:
                connection.close()
    
    def _check_system_resources(self) -> HealthCheck:
        """Check system resource utilization"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on resource usage
            status = "healthy"
            messages = []
            
            if cpu_percent > 90:
                status = "critical"
                messages.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent > 70:
                status = "warning"
                messages.append(f"CPU usage high: {cpu_percent:.1f}%")
            
            if memory.percent > 90:
                status = "critical"
                messages.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent > 80:
                if status != "critical":
                    status = "warning"
                messages.append(f"Memory usage high: {memory.percent:.1f}%")
            
            disk_percent = (disk.used / disk.total) * 100
            if disk_percent > 90:
                status = "critical"
                messages.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent > 80:
                if status != "critical":
                    status = "warning"
                messages.append(f"Disk usage high: {disk_percent:.1f}%")
            
            message = "; ".join(messages) if messages else "System resources within normal ranges"
            
            return HealthCheck(
                check_name="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "disk_percent": disk_percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="system_resources",
                status="critical",
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now(),
                metrics={}
            )
    
    def _check_chunk_distribution(self) -> HealthCheck:
        """Check chunk access distribution and identify issues"""
        connection = None
        try:
            connection = self.metrics_collector.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Get chunk statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(CASE WHEN access_count = 0 THEN 1 END) as never_accessed,
                    COUNT(CASE WHEN access_count BETWEEN 1 AND 5 THEN 1 END) as low_access,
                    COUNT(CASE WHEN access_count > 50 THEN 1 END) as high_access,
                    AVG(access_count) as avg_access,
                    MAX(access_count) as max_access
                FROM megamind_chunks
            """)
            
            stats = cursor.fetchone()
            
            # Calculate distribution metrics
            total_chunks = stats['total_chunks']
            never_accessed_pct = (stats['never_accessed'] / total_chunks) * 100 if total_chunks > 0 else 0
            low_access_pct = (stats['low_access'] / total_chunks) * 100 if total_chunks > 0 else 0
            
            # Determine status
            status = "healthy"
            messages = []
            
            if never_accessed_pct > 50:
                status = "warning"
                messages.append(f"High percentage of never-accessed chunks: {never_accessed_pct:.1f}%")
            
            if low_access_pct > 70:
                status = "warning"
                messages.append(f"High percentage of low-access chunks: {low_access_pct:.1f}%")
            
            message = "; ".join(messages) if messages else "Chunk access distribution is healthy"
            
            return HealthCheck(
                check_name="chunk_distribution",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    "total_chunks": float(total_chunks),
                    "never_accessed_percent": never_accessed_pct,
                    "low_access_percent": low_access_pct,
                    "avg_access": float(stats['avg_access'] or 0),
                    "max_access": float(stats['max_access'] or 0)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="chunk_distribution",
                status="critical",
                message=f"Failed to check chunk distribution: {str(e)}",
                timestamp=datetime.now(),
                metrics={}
            )
        finally:
            if connection:
                connection.close()
    
    def _check_relationship_integrity(self) -> HealthCheck:
        """Check relationship data integrity"""
        connection = None
        try:
            connection = self.metrics_collector.get_connection()
            cursor = connection.cursor(dictionary=True)
            
            # Check for orphaned relationships
            cursor.execute("""
                SELECT COUNT(*) as orphaned_relationships
                FROM megamind_chunk_relationships r
                LEFT JOIN megamind_chunks c1 ON r.chunk_id = c1.chunk_id
                LEFT JOIN megamind_chunks c2 ON r.related_chunk_id = c2.chunk_id
                WHERE c1.chunk_id IS NULL OR c2.chunk_id IS NULL
            """)
            
            orphaned_count = cursor.fetchone()['orphaned_relationships']
            
            # Check relationship distribution
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_relationships,
                    COUNT(DISTINCT chunk_id) as chunks_with_relationships,
                    AVG(strength) as avg_strength
                FROM megamind_chunk_relationships
            """)
            
            rel_stats = cursor.fetchone()
            
            # Determine status
            status = "healthy"
            messages = []
            
            if orphaned_count > 0:
                status = "warning"
                messages.append(f"Found {orphaned_count} orphaned relationships")
            
            if rel_stats['total_relationships'] == 0:
                status = "warning"
                messages.append("No relationships found in database")
            
            message = "; ".join(messages) if messages else "Relationship integrity is good"
            
            return HealthCheck(
                check_name="relationship_integrity",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    "total_relationships": float(rel_stats['total_relationships'] or 0),
                    "chunks_with_relationships": float(rel_stats['chunks_with_relationships'] or 0),
                    "orphaned_relationships": float(orphaned_count),
                    "avg_strength": float(rel_stats['avg_strength'] or 0)
                }
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="relationship_integrity",
                status="critical",
                message=f"Failed to check relationship integrity: {str(e)}",
                timestamp=datetime.now(),
                metrics={}
            )
        finally:
            if connection:
                connection.close()
    
    def _check_performance_metrics(self) -> HealthCheck:
        """Check system performance metrics"""
        try:
            # Get recent metrics summary
            cpu_summary = self.metrics_collector.get_metric_summary("system_cpu_percent", 30)
            memory_summary = self.metrics_collector.get_metric_summary("system_memory_percent", 30)
            
            # Check if we have recent data
            if not cpu_summary or not memory_summary:
                return HealthCheck(
                    check_name="performance_metrics",
                    status="warning",
                    message="Insufficient recent performance data",
                    timestamp=datetime.now(),
                    metrics={}
                )
            
            # Analyze performance trends
            status = "healthy"
            messages = []
            
            if cpu_summary['avg'] > 70:
                status = "warning"
                messages.append(f"Average CPU usage high: {cpu_summary['avg']:.1f}%")
            
            if memory_summary['avg'] > 80:
                status = "warning"
                messages.append(f"Average memory usage high: {memory_summary['avg']:.1f}%")
            
            message = "; ".join(messages) if messages else "Performance metrics are within acceptable ranges"
            
            return HealthCheck(
                check_name="performance_metrics",
                status=status,
                message=message,
                timestamp=datetime.now(),
                metrics={
                    "cpu_avg": cpu_summary['avg'],
                    "cpu_max": cpu_summary['max'],
                    "memory_avg": memory_summary['avg'],
                    "memory_max": memory_summary['max']
                }
            )
            
        except Exception as e:
            return HealthCheck(
                check_name="performance_metrics",
                status="critical",
                message=f"Failed to check performance metrics: {str(e)}",
                timestamp=datetime.now(),
                metrics={}
            )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health_checks = self.run_health_checks()
        
        # Determine overall status
        statuses = [check.status for check in health_checks]
        if "critical" in statuses:
            overall_status = "critical"
        elif "warning" in statuses:
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        # Get recent metrics
        recent_metrics = {
            "cpu": self.metrics_collector.get_metric_summary("system_cpu_percent", 5),
            "memory": self.metrics_collector.get_metric_summary("system_memory_percent", 5),
            "chunks": self.metrics_collector.get_metric_summary("chunks_total", 5),
            "relationships": self.metrics_collector.get_metric_summary("relationships_total", 5),
        }
        
        # Get active alerts
        active_alerts = [asdict(alert) for alert in self.alert_manager.active_alerts.values()]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "health_checks": [asdict(check) for check in health_checks],
            "recent_metrics": recent_metrics,
            "active_alerts": active_alerts,
            "alert_summary": {
                "total_active": len(active_alerts),
                "critical": len([a for a in active_alerts if a['severity'] == 'critical']),
                "warning": len([a for a in active_alerts if a['severity'] == 'warning']),
                "info": len([a for a in active_alerts if a['severity'] == 'info'])
            }
        }

def load_config():
    """Load configuration from environment variables"""
    return {
        'host': os.getenv('MEGAMIND_DB_HOST', '10.255.250.22'),
        'port': os.getenv('MEGAMIND_DB_PORT', '3309'),
        'database': os.getenv('MEGAMIND_DB_NAME', 'megamind_database'),
        'user': os.getenv('MEGAMIND_DB_USER', 'megamind_user'),
        'password': os.getenv('MEGAMIND_DB_PASSWORD', ''),
    }

def main():
    """Main entry point for system health monitoring"""
    try:
        # Load configuration
        db_config = load_config()
        
        if not db_config['password']:
            logger.error("Database password not configured. Set MEGAMIND_DB_PASSWORD environment variable.")
            return 1
        
        # Initialize monitoring system
        monitor = SystemHealthMonitor(db_config)
        
        # Start monitoring
        monitor.start()
        
        try:
            # Run for demonstration
            logger.info("System health monitoring running... Press Ctrl+C to stop")
            
            # Generate initial health report
            status = monitor.get_system_status()
            
            print("\n" + "="*60)
            print("MEGAMIND SYSTEM HEALTH REPORT")
            print("="*60)
            print(f"Overall Status: {status['overall_status'].upper()}")
            print(f"Timestamp: {status['timestamp']}")
            print(f"Active Alerts: {status['alert_summary']['total_active']}")
            
            print("\nHealth Checks:")
            for check in status['health_checks']:
                status_icon = "✓" if check['status'] == 'healthy' else "⚠" if check['status'] == 'warning' else "✗"
                print(f"  {status_icon} {check['check_name']}: {check['status']} - {check['message']}")
            
            print("\nRecent Metrics:")
            for metric_name, summary in status['recent_metrics'].items():
                if summary:
                    print(f"  {metric_name}: latest={summary.get('latest', 0):.2f}, avg={summary.get('avg', 0):.2f}")
            
            if status['active_alerts']:
                print(f"\nActive Alerts ({len(status['active_alerts'])}):")
                for alert in status['active_alerts']:
                    print(f"  [{alert['severity'].upper()}] {alert['metric_name']}: {alert['message']}")
            
            # Keep running until interrupted
            while True:
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            monitor.stop()
        
        return 0
        
    except Exception as e:
        logger.error(f"System health monitoring failed: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())