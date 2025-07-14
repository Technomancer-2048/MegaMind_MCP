#!/usr/bin/env python3
"""
Dynamic Realm Audit Logger
Provides comprehensive audit logging for all dynamic realm operations
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    REALM_CREATION = "realm_creation"
    REALM_ACCESS = "realm_access"
    REALM_VALIDATION = "realm_validation"
    REALM_CONFIGURATION = "realm_configuration"
    SECURITY_VIOLATION = "security_violation"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHECK = "permission_check"
    CONFIGURATION_CHANGE = "configuration_change"
    SYSTEM_EVENT = "system_event"

class AuditSeverity(Enum):
    """Severity levels for audit events"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AuditEvent:
    """Individual audit event record"""
    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    realm_id: str
    operation: str
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    request_context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['event_type'] = self.event_type.value
        result['severity'] = self.severity.value
        return result
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

class DynamicRealmAuditLogger:
    """Comprehensive audit logging for dynamic realm operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Audit configuration
        self.audit_enabled = self.config.get('audit_enabled', True)
        self.log_to_file = self.config.get('log_to_file', True)
        self.log_to_syslog = self.config.get('log_to_syslog', False)
        self.log_to_database = self.config.get('log_to_database', False)
        
        # File logging configuration
        self.audit_log_path = self.config.get('audit_log_path', '/var/log/megamind/audit.log')
        self.max_log_size_mb = self.config.get('max_log_size_mb', 100)
        self.max_log_files = self.config.get('max_log_files', 10)
        
        # Security and compliance settings
        self.mask_sensitive_data = self.config.get('mask_sensitive_data', True)
        self.retention_days = self.config.get('retention_days', 90)
        self.compliance_format = self.config.get('compliance_format', 'iso27001')
        
        # Initialize logging infrastructure
        self._setup_audit_logging()
        
        # Event buffer for high-throughput scenarios
        self._event_buffer: List[AuditEvent] = []
        self._buffer_size = self.config.get('buffer_size', 100)
        self._buffer_flush_interval = self.config.get('buffer_flush_interval', 60)  # seconds
        
        logger.info("DynamicRealmAuditLogger initialized with comprehensive audit capabilities")
    
    def _setup_audit_logging(self):
        """Setup audit logging infrastructure"""
        try:
            if self.log_to_file:
                # Ensure audit log directory exists
                log_dir = Path(self.audit_log_path).parent
                log_dir.mkdir(parents=True, exist_ok=True)
                
                # Setup file handler with rotation
                from logging.handlers import RotatingFileHandler
                
                audit_handler = RotatingFileHandler(
                    self.audit_log_path,
                    maxBytes=self.max_log_size_mb * 1024 * 1024,
                    backupCount=self.max_log_files
                )
                
                # Audit-specific formatter
                audit_formatter = logging.Formatter(
                    '%(asctime)s - AUDIT - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S UTC'
                )
                audit_handler.setFormatter(audit_formatter)
                
                # Create audit-specific logger
                self.audit_logger = logging.getLogger('megamind.audit')
                self.audit_logger.addHandler(audit_handler)
                self.audit_logger.setLevel(logging.INFO)
                
                logger.info(f"Audit file logging enabled: {self.audit_log_path}")
            
            if self.log_to_syslog:
                self._setup_syslog_logging()
                
        except Exception as e:
            logger.error(f"Failed to setup audit logging: {e}")
            self.audit_enabled = False
    
    def _setup_syslog_logging(self):
        """Setup syslog logging for audit events"""
        try:
            from logging.handlers import SysLogHandler
            
            syslog_handler = SysLogHandler(address='/dev/log')
            syslog_formatter = logging.Formatter(
                'megamind-audit[%(process)d]: %(message)s'
            )
            syslog_handler.setFormatter(syslog_formatter)
            
            if hasattr(self, 'audit_logger'):
                self.audit_logger.addHandler(syslog_handler)
            
            logger.info("Audit syslog logging enabled")
            
        except Exception as e:
            logger.warning(f"Failed to setup syslog logging: {e}")
    
    def log_realm_creation(self, realm_id: str, realm_config: Dict[str, Any], 
                          request_context: Optional[Dict[str, Any]] = None) -> str:
        """Log dynamic realm creation event"""
        try:
            details = {
                'realm_config': self._mask_sensitive_data(realm_config) if self.mask_sensitive_data else realm_config,
                'config_hash': self._hash_config(realm_config),
                'creation_method': 'dynamic_header'
            }
            
            event = self._create_audit_event(
                event_type=AuditEventType.REALM_CREATION,
                severity=AuditSeverity.INFO,
                realm_id=realm_id,
                operation="create_dynamic_realm",
                success=True,
                message=f"Dynamic realm created: {realm_id}",
                details=details,
                request_context=request_context
            )
            
            self._log_audit_event(event)
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to log realm creation: {e}")
            return ""
    
    def log_realm_access(self, realm_id: str, operation: str, success: bool, 
                        reason: str = "", details: Optional[Dict[str, Any]] = None,
                        request_context: Optional[Dict[str, Any]] = None) -> str:
        """Log realm access attempt"""
        try:
            severity = AuditSeverity.INFO if success else AuditSeverity.WARNING
            
            audit_details = {
                'access_reason': reason,
                'access_method': details.get('access_method', 'unknown') if details else 'unknown',
                'resource_accessed': details.get('resource', 'unknown') if details else 'unknown'
            }
            
            if details:
                audit_details.update(details)
            
            event = self._create_audit_event(
                event_type=AuditEventType.REALM_ACCESS,
                severity=severity,
                realm_id=realm_id,
                operation=operation,
                success=success,
                message=f"Realm access {'granted' if success else 'denied'}: {realm_id} - {operation}",
                details=audit_details,
                request_context=request_context
            )
            
            self._log_audit_event(event)
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to log realm access: {e}")
            return ""
    
    def log_security_violation(self, realm_id: str, violation_type: str, 
                             violation_details: Dict[str, Any],
                             request_context: Optional[Dict[str, Any]] = None) -> str:
        """Log security violation"""
        try:
            details = {
                'violation_type': violation_type,
                'violation_data': self._mask_sensitive_data(violation_details) if self.mask_sensitive_data else violation_details,
                'security_level': 'high',
                'requires_investigation': True
            }
            
            event = self._create_audit_event(
                event_type=AuditEventType.SECURITY_VIOLATION,
                severity=AuditSeverity.CRITICAL,
                realm_id=realm_id,
                operation="security_check",
                success=False,
                message=f"Security violation detected: {violation_type} in realm {realm_id}",
                details=details,
                request_context=request_context
            )
            
            self._log_audit_event(event)
            
            # Also log to application logger for immediate attention
            logger.critical(f"SECURITY VIOLATION: {violation_type} in realm {realm_id} - Event ID: {event.event_id}")
            
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to log security violation: {e}")
            return ""
    
    def log_configuration_validation(self, realm_id: str, validation_results: List[Dict[str, Any]],
                                   config_hash: str, request_context: Optional[Dict[str, Any]] = None) -> str:
        """Log configuration validation results"""
        try:
            # Determine overall success and severity
            has_errors = any(r.get('severity') in ['error', 'critical'] for r in validation_results)
            has_warnings = any(r.get('severity') == 'warning' for r in validation_results)
            
            success = not has_errors
            severity = AuditSeverity.ERROR if has_errors else (AuditSeverity.WARNING if has_warnings else AuditSeverity.INFO)
            
            details = {
                'validation_results': validation_results,
                'config_hash': config_hash,
                'total_checks': len(validation_results),
                'errors': sum(1 for r in validation_results if r.get('severity') in ['error', 'critical']),
                'warnings': sum(1 for r in validation_results if r.get('severity') == 'warning')
            }
            
            event = self._create_audit_event(
                event_type=AuditEventType.REALM_VALIDATION,
                severity=severity,
                realm_id=realm_id,
                operation="validate_configuration",
                success=success,
                message=f"Configuration validation {'passed' if success else 'failed'}: {realm_id}",
                details=details,
                request_context=request_context
            )
            
            self._log_audit_event(event)
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to log configuration validation: {e}")
            return ""
    
    def log_permission_check(self, realm_id: str, permission_type: str, resource: str,
                           granted: bool, reason: str = "",
                           request_context: Optional[Dict[str, Any]] = None) -> str:
        """Log permission check"""
        try:
            details = {
                'permission_type': permission_type,
                'resource': resource,
                'check_reason': reason,
                'permission_level': request_context.get('permission_level', 'unknown') if request_context else 'unknown'
            }
            
            event = self._create_audit_event(
                event_type=AuditEventType.PERMISSION_CHECK,
                severity=AuditSeverity.INFO if granted else AuditSeverity.WARNING,
                realm_id=realm_id,
                operation=f"check_{permission_type}_permission",
                success=granted,
                message=f"Permission {'granted' if granted else 'denied'}: {permission_type} on {resource}",
                details=details,
                request_context=request_context
            )
            
            self._log_audit_event(event)
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to log permission check: {e}")
            return ""
    
    def log_system_event(self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None,
                        severity: AuditSeverity = AuditSeverity.INFO) -> str:
        """Log system-level audit event"""
        try:
            event = self._create_audit_event(
                event_type=AuditEventType.SYSTEM_EVENT,
                severity=severity,
                realm_id="SYSTEM",
                operation=event_type,
                success=True,
                message=message,
                details=details or {}
            )
            
            self._log_audit_event(event)
            return event.event_id
            
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
            return ""
    
    def _create_audit_event(self, event_type: AuditEventType, severity: AuditSeverity,
                           realm_id: str, operation: str, success: bool, message: str,
                           details: Optional[Dict[str, Any]] = None,
                           request_context: Optional[Dict[str, Any]] = None) -> AuditEvent:
        """Create audit event with standard metadata"""
        
        # Extract user context from request if available
        user_context = {}
        if request_context:
            user_context = {
                'client_ip': request_context.get('client_ip', 'unknown'),
                'user_agent': request_context.get('user_agent', 'unknown'),
                'session_id': request_context.get('session_id', 'unknown'),
                'request_id': request_context.get('request_id', 'unknown')
            }
        
        # Enhance details with system context
        enhanced_details = details or {}
        enhanced_details.update({
            'hostname': os.getenv('HOSTNAME', 'unknown'),
            'container_id': os.getenv('CONTAINER_ID', 'unknown'),
            'process_id': os.getpid(),
            'audit_version': '1.0'
        })
        
        return AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            realm_id=realm_id,
            operation=operation,
            success=success,
            message=message,
            details=enhanced_details,
            user_context=user_context,
            request_context=request_context
        )
    
    def _log_audit_event(self, event: AuditEvent):
        """Log audit event to configured destinations"""
        if not self.audit_enabled:
            return
        
        try:
            # Format audit message
            audit_message = self._format_audit_message(event)
            
            # Log to file
            if hasattr(self, 'audit_logger'):
                if event.severity in [AuditSeverity.CRITICAL, AuditSeverity.ERROR]:
                    self.audit_logger.error(audit_message)
                elif event.severity == AuditSeverity.WARNING:
                    self.audit_logger.warning(audit_message)
                else:
                    self.audit_logger.info(audit_message)
            
            # Add to buffer for batch processing
            self._event_buffer.append(event)
            
            # Flush buffer if full
            if len(self._event_buffer) >= self._buffer_size:
                self._flush_event_buffer()
                
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
    
    def _format_audit_message(self, event: AuditEvent) -> str:
        """Format audit event for logging"""
        if self.compliance_format == 'iso27001':
            return self._format_iso27001(event)
        elif self.compliance_format == 'json':
            return event.to_json()
        else:
            return self._format_standard(event)
    
    def _format_iso27001(self, event: AuditEvent) -> str:
        """Format audit event for ISO 27001 compliance"""
        return (f"[{event.timestamp.isoformat()}] "
                f"ID={event.event_id} "
                f"TYPE={event.event_type.value} "
                f"SEVERITY={event.severity.value} "
                f"REALM={event.realm_id} "
                f"OP={event.operation} "
                f"SUCCESS={event.success} "
                f"MSG=\"{event.message}\" "
                f"CLIENT_IP={event.user_context.get('client_ip', 'unknown') if event.user_context else 'unknown'}")
    
    def _format_standard(self, event: AuditEvent) -> str:
        """Format audit event in standard format"""
        return (f"{event.event_type.value.upper()} - "
                f"{event.realm_id} - "
                f"{event.operation} - "
                f"{'SUCCESS' if event.success else 'FAILURE'} - "
                f"{event.message}")
    
    def _mask_sensitive_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mask sensitive data in audit logs"""
        if not isinstance(data, dict):
            return data
        
        masked_data = data.copy()
        sensitive_keys = {'password', 'secret', 'token', 'key', 'credential', 'auth'}
        
        for key, value in masked_data.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                if isinstance(value, str) and len(value) > 4:
                    masked_data[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    masked_data[key] = '***'
            elif isinstance(value, dict):
                masked_data[key] = self._mask_sensitive_data(value)
        
        return masked_data
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration tracking"""
        import hashlib
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def _flush_event_buffer(self):
        """Flush event buffer to persistent storage"""
        if not self._event_buffer:
            return
        
        try:
            # For database logging (if enabled)
            if self.log_to_database:
                self._write_events_to_database(self._event_buffer)
            
            # Clear buffer
            self._event_buffer.clear()
            logger.debug(f"Flushed {len(self._event_buffer)} audit events")
            
        except Exception as e:
            logger.error(f"Failed to flush audit event buffer: {e}")
    
    def _write_events_to_database(self, events: List[AuditEvent]):
        """Write audit events to database (placeholder for database integration)"""
        # This would integrate with the MegaMind database to store audit events
        # in a dedicated audit table
        pass
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for the last N hours"""
        # This would query audit logs and provide summary statistics
        # Implementation depends on the storage backend
        return {
            'period_hours': hours,
            'total_events': 0,  # Placeholder
            'security_violations': 0,
            'access_denials': 0,
            'realm_creations': 0,
            'unique_realms': 0
        }
    
    def search_audit_events(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search audit events by criteria"""
        # This would implement audit log search functionality
        # Implementation depends on the storage backend
        return []
    
    def cleanup_old_events(self):
        """Clean up audit events older than retention period"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
            # Implementation would remove events older than cutoff_date
            logger.info(f"Audit cleanup: removing events older than {cutoff_date}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old audit events: {e}")
    
    def shutdown(self):
        """Shutdown audit logger and flush remaining events"""
        try:
            self._flush_event_buffer()
            logger.info("Audit logger shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during audit logger shutdown: {e}")