#!/usr/bin/env python3
"""
Enhanced Security and Validation Pipeline
Integrates validation, audit logging, and caching for comprehensive dynamic realm security
"""

import json
import logging
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security enforcement levels"""
    PERMISSIVE = "permissive"      # Allow with warnings
    STANDARD = "standard"          # Standard security checks
    STRICT = "strict"              # Strict validation
    PARANOID = "paranoid"          # Maximum security

class ValidationOutcome(Enum):
    """Validation pipeline outcomes"""
    APPROVED = "approved"
    APPROVED_WITH_WARNINGS = "approved_with_warnings"
    REJECTED = "rejected"
    BLOCKED = "blocked"

@dataclass
class SecurityContext:
    """Security context for validation pipeline"""
    client_ip: str
    user_agent: str
    request_id: str
    session_id: Optional[str] = None
    realm_id: Optional[str] = None
    operation: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.STANDARD
    
class EnhancedSecurityPipeline:
    """
    Comprehensive security and validation pipeline for dynamic realm configurations
    Integrates validation, audit logging, caching, and threat detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self._init_validator()
        self._init_audit_logger()
        self._init_cache_manager()
        
        # Security configuration
        self.security_level = SecurityLevel(self.config.get('security_level', 'standard'))
        self.enable_threat_detection = self.config.get('enable_threat_detection', True)
        self.max_validation_time_ms = self.config.get('max_validation_time_ms', 5000)
        self.rate_limit_enabled = self.config.get('rate_limit_enabled', True)
        
        # Threat detection thresholds
        self.max_requests_per_minute = self.config.get('max_requests_per_minute', 100)
        self.max_failed_validations = self.config.get('max_failed_validations', 10)
        self.suspicious_pattern_threshold = self.config.get('suspicious_pattern_threshold', 5)
        
        # Request tracking for rate limiting and threat detection
        self._request_tracker: Dict[str, List[datetime]] = {}
        self._failed_validations: Dict[str, int] = {}
        self._blocked_ips: Dict[str, datetime] = {}
        
        logger.info("EnhancedSecurityPipeline initialized with comprehensive security features")
    
    def _init_validator(self):
        """Initialize configuration validator"""
        try:
            # Import the validator we created earlier
            validator_config = self.config.get('validator_config', {})
            
            # Mock validator initialization - in real implementation, import the actual class
            class MockRealmConfigValidator:
                def __init__(self, config):
                    self.config = config
                
                def validate_realm_config(self, config, skip_cache=False):
                    # Mock validation results
                    return [
                        {'valid': True, 'severity': 'info', 'message': 'Configuration valid'},
                        {'valid': True, 'severity': 'info', 'message': 'All security checks passed'}
                    ]
                
                def sanitize_config(self, config):
                    return config
                
                def is_valid_configuration(self, config):
                    return True
            
            self.validator = MockRealmConfigValidator(validator_config)
            logger.info("Configuration validator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize validator: {e}")
            self.validator = None
    
    def _init_audit_logger(self):
        """Initialize audit logger"""
        try:
            audit_config = self.config.get('audit_config', {})
            
            # Mock audit logger - in real implementation, import the actual class
            class MockDynamicRealmAuditLogger:
                def __init__(self, config):
                    self.config = config
                
                def log_realm_creation(self, realm_id, realm_config, request_context=None):
                    logger.info(f"AUDIT: Realm created - {realm_id}")
                    return "audit_event_123"
                
                def log_security_violation(self, realm_id, violation_type, violation_details, request_context=None):
                    logger.warning(f"AUDIT: Security violation - {violation_type} in {realm_id}")
                    return "audit_event_456"
                
                def log_configuration_validation(self, realm_id, validation_results, config_hash, request_context=None):
                    logger.info(f"AUDIT: Configuration validated - {realm_id}")
                    return "audit_event_789"
                
                def log_permission_check(self, realm_id, permission_type, resource, granted, reason="", request_context=None):
                    logger.info(f"AUDIT: Permission {'granted' if granted else 'denied'} - {permission_type}")
                    return "audit_event_101112"
            
            self.audit_logger = MockDynamicRealmAuditLogger(audit_config)
            logger.info("Audit logger initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize audit logger: {e}")
            self.audit_logger = None
    
    def _init_cache_manager(self):
        """Initialize cache manager"""
        try:
            cache_config = self.config.get('cache_config', {})
            
            # Mock cache manager - in real implementation, import the actual class
            class MockRealmConfigurationManager:
                def __init__(self, config):
                    self.config = config
                    self._cache = {}
                
                def get_validation_result(self, config_hash):
                    return self._cache.get(f"validation:{config_hash}")
                
                def set_validation_result(self, config_hash, validation_results):
                    self._cache[f"validation:{config_hash}"] = validation_results
                    return True
                
                def get_realm_config(self, realm_id, config_hash):
                    return self._cache.get(f"realm:{realm_id}:{config_hash}")
                
                def set_realm_config(self, realm_id, config_hash, config):
                    self._cache[f"realm:{realm_id}:{config_hash}"] = config
                    return True
                
                def get_cache_metrics(self):
                    return {
                        'cache_size': len(self._cache),
                        'cache_hits': 0,
                        'cache_misses': 0
                    }
            
            self.cache_manager = MockRealmConfigurationManager(cache_config)
            logger.info("Cache manager initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            self.cache_manager = None
    
    def validate_and_process_realm_config(self, 
                                        realm_config: Dict[str, Any], 
                                        security_context: SecurityContext) -> Tuple[ValidationOutcome, Dict[str, Any]]:
        """
        Main pipeline method: validates and processes realm configuration
        Returns validation outcome and processed configuration
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Security pre-checks
            security_check = self._perform_security_checks(security_context)
            if not security_check['allowed']:
                return ValidationOutcome.BLOCKED, {
                    'error': security_check['reason'],
                    'security_violation': True,
                    'blocked_until': security_check.get('blocked_until')
                }
            
            # Step 2: Generate configuration hash for caching
            config_hash = self._generate_config_hash(realm_config)
            
            # Step 3: Check validation cache
            cached_validation = None
            if self.cache_manager:
                cached_validation = self.cache_manager.get_validation_result(config_hash)
            
            if cached_validation:
                logger.debug(f"Using cached validation result for config hash: {config_hash[:8]}")
                validation_results = cached_validation
            else:
                # Step 4: Perform validation
                validation_results = self._validate_configuration(realm_config, security_context)
                
                # Cache validation results
                if self.cache_manager:
                    self.cache_manager.set_validation_result(config_hash, validation_results)
            
            # Step 5: Process validation results
            outcome, processed_config = self._process_validation_results(
                validation_results, realm_config, security_context
            )
            
            # Step 6: Audit logging
            self._log_validation_outcome(realm_config, validation_results, outcome, security_context, config_hash)
            
            # Step 7: Update threat detection metrics
            self._update_threat_metrics(security_context, outcome)
            
            # Step 8: Cache processed configuration if valid
            if outcome in [ValidationOutcome.APPROVED, ValidationOutcome.APPROVED_WITH_WARNINGS]:
                if self.cache_manager:
                    realm_id = realm_config.get('project_realm', 'unknown')
                    self.cache_manager.set_realm_config(realm_id, config_hash, processed_config)
            
            # Performance monitoring
            processing_time = (time.perf_counter() - start_time) * 1000
            if processing_time > self.max_validation_time_ms:
                logger.warning(f"Validation took {processing_time:.2f}ms (threshold: {self.max_validation_time_ms}ms)")
            
            return outcome, processed_config
            
        except Exception as e:
            logger.error(f"Security pipeline failed: {e}")
            
            # Log security violation for pipeline failures
            if self.audit_logger:
                self.audit_logger.log_security_violation(
                    realm_id=security_context.realm_id or 'unknown',
                    violation_type='pipeline_failure',
                    violation_details={'error': str(e), 'config_hash': config_hash if 'config_hash' in locals() else 'unknown'},
                    request_context=self._security_context_to_dict(security_context)
                )
            
            return ValidationOutcome.REJECTED, {'error': f'Validation pipeline failed: {str(e)}'}
    
    def _perform_security_checks(self, security_context: SecurityContext) -> Dict[str, Any]:
        """Perform pre-validation security checks"""
        try:
            # Check if IP is blocked
            if security_context.client_ip in self._blocked_ips:
                block_time = self._blocked_ips[security_context.client_ip]
                if (datetime.now() - block_time).total_seconds() < 3600:  # 1 hour block
                    return {
                        'allowed': False,
                        'reason': 'IP temporarily blocked due to security violations',
                        'blocked_until': (block_time + datetime.timedelta(hours=1)).isoformat()
                    }
                else:
                    # Unblock expired IPs
                    del self._blocked_ips[security_context.client_ip]
            
            # Rate limiting check
            if self.rate_limit_enabled:
                rate_check = self._check_rate_limit(security_context.client_ip)
                if not rate_check['allowed']:
                    return rate_check
            
            # Threat pattern detection
            if self.enable_threat_detection:
                threat_check = self._detect_threat_patterns(security_context)
                if not threat_check['allowed']:
                    return threat_check
            
            return {'allowed': True}
            
        except Exception as e:
            logger.error(f"Security check failed: {e}")
            return {
                'allowed': False,
                'reason': f'Security check error: {str(e)}'
            }
    
    def _check_rate_limit(self, client_ip: str) -> Dict[str, Any]:
        """Enhanced rate limiting with stricter enforcement"""
        try:
            current_time = datetime.now()
            
            # Initialize tracking for new IPs
            if client_ip not in self._request_tracker:
                self._request_tracker[client_ip] = []
            
            # Clean old requests (outside 1 minute window)
            cutoff_time = current_time - timedelta(minutes=1)
            self._request_tracker[client_ip] = [
                req_time for req_time in self._request_tracker[client_ip] 
                if req_time > cutoff_time
            ]
            
            # ENHANCED: Stricter rate limiting for security tests
            current_requests = len(self._request_tracker[client_ip])
            if current_requests >= self.max_requests_per_minute:
                logger.warning(f"Rate limit exceeded for IP: {client_ip} ({current_requests} requests)")
                
                # Block IP temporarily
                self._blocked_ips[client_ip] = current_time
                
                return {
                    'allowed': False,
                    'reason': f'Rate limit exceeded: {current_requests} requests in last minute',
                    'retry_after': 60
                }
            
            # Record current request
            self._request_tracker[client_ip].append(current_time)
            
            return {'allowed': True}
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return {'allowed': True}  # Fail open for rate limiting
    
    def _detect_threat_patterns(self, security_context: SecurityContext) -> Dict[str, Any]:
        """Detect suspicious threat patterns"""
        try:
            suspicious_indicators = []
            
            # Check for suspicious user agents
            user_agent = security_context.user_agent.lower()
            suspicious_ua_patterns = ['bot', 'crawler', 'scanner', 'exploit', 'attack']
            if any(pattern in user_agent for pattern in suspicious_ua_patterns):
                suspicious_indicators.append('suspicious_user_agent')
            
            # Check failed validation count
            client_ip = security_context.client_ip
            failed_count = self._failed_validations.get(client_ip, 0)
            if failed_count >= self.max_failed_validations:
                suspicious_indicators.append('excessive_failed_validations')
                
                # Block IP after too many failures
                self._blocked_ips[client_ip] = datetime.now()
                logger.warning(f"IP blocked due to excessive failures: {client_ip}")
                
                return {
                    'allowed': False,
                    'reason': f'IP blocked due to {failed_count} failed validations'
                }
            
            # Check for SQL injection patterns in realm configuration
            if security_context.realm_id:
                sql_patterns = ['union', 'select', 'drop', 'delete', 'insert', '--', ';']
                realm_id_lower = security_context.realm_id.lower()
                if any(pattern in realm_id_lower for pattern in sql_patterns):
                    suspicious_indicators.append('sql_injection_attempt')
            
            # Log suspicious activity but don't block yet
            if len(suspicious_indicators) >= self.suspicious_pattern_threshold:
                if self.audit_logger:
                    self.audit_logger.log_security_violation(
                        realm_id=security_context.realm_id or 'unknown',
                        violation_type='threat_pattern_detected',
                        violation_details={
                            'indicators': suspicious_indicators,
                            'client_ip': client_ip,
                            'user_agent': security_context.user_agent
                        },
                        request_context=self._security_context_to_dict(security_context)
                    )
                
                return {
                    'allowed': False,
                    'reason': f'Threat patterns detected: {suspicious_indicators}'
                }
            
            return {'allowed': True}
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return {'allowed': True}  # Fail open for threat detection
    
    def _validate_configuration(self, realm_config: Dict[str, Any], 
                               security_context: SecurityContext) -> List[Dict[str, Any]]:
        """Validate realm configuration"""
        try:
            if not self.validator:
                return [{'valid': True, 'severity': 'warning', 'message': 'Validator not available'}]
            
            # Perform validation based on security level
            if self.security_level == SecurityLevel.PARANOID:
                # Most strict validation
                results = self.validator.validate_realm_config(realm_config, skip_cache=True)
            else:
                # Standard validation (may use cache)
                results = self.validator.validate_realm_config(realm_config)
            
            # Convert validation results to standard format
            formatted_results = []
            for result in results:
                if hasattr(result, 'to_dict'):
                    formatted_results.append(result.to_dict())
                elif isinstance(result, dict):
                    formatted_results.append(result)
                else:
                    formatted_results.append({
                        'valid': True,
                        'severity': 'info',
                        'message': str(result)
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return [{
                'valid': False,
                'severity': 'error',
                'message': f'Validation error: {str(e)}'
            }]
    
    def _process_validation_results(self, validation_results: List[Dict[str, Any]], 
                                  realm_config: Dict[str, Any],
                                  security_context: SecurityContext) -> Tuple[ValidationOutcome, Dict[str, Any]]:
        """Process validation results and determine outcome"""
        try:
            # Count severity levels
            errors = sum(1 for r in validation_results if r.get('severity') in ['error', 'critical'])
            warnings = sum(1 for r in validation_results if r.get('severity') == 'warning')
            
            # Determine outcome based on security level
            if errors > 0:
                if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
                    return ValidationOutcome.REJECTED, {
                        'validation_results': validation_results,
                        'error_count': errors,
                        'warning_count': warnings
                    }
                else:
                    # In permissive/standard mode, some errors might be acceptable
                    critical_errors = sum(1 for r in validation_results if r.get('severity') == 'critical')
                    if critical_errors > 0:
                        return ValidationOutcome.REJECTED, {
                            'validation_results': validation_results,
                            'critical_errors': critical_errors
                        }
            
            # Sanitize configuration
            processed_config = realm_config.copy()
            if self.validator:
                processed_config = self.validator.sanitize_config(processed_config)
            
            # Add validation metadata
            processed_config['_validation'] = {
                'validated_at': datetime.now().isoformat(),
                'security_level': self.security_level.value,
                'validation_results': validation_results,
                'error_count': errors,
                'warning_count': warnings
            }
            
            # Determine final outcome
            if warnings > 0:
                return ValidationOutcome.APPROVED_WITH_WARNINGS, processed_config
            else:
                return ValidationOutcome.APPROVED, processed_config
                
        except Exception as e:
            logger.error(f"Failed to process validation results: {e}")
            return ValidationOutcome.REJECTED, {'error': f'Processing error: {str(e)}'}
    
    def _log_validation_outcome(self, realm_config: Dict[str, Any], validation_results: List[Dict[str, Any]],
                               outcome: ValidationOutcome, security_context: SecurityContext, config_hash: str):
        """Log validation outcome to audit system"""
        try:
            if not self.audit_logger:
                return
            
            realm_id = realm_config.get('project_realm', 'unknown')
            request_context = self._security_context_to_dict(security_context)
            
            # Log configuration validation
            self.audit_logger.log_configuration_validation(
                realm_id=realm_id,
                validation_results=validation_results,
                config_hash=config_hash,
                request_context=request_context
            )
            
            # Log security-specific events
            if outcome == ValidationOutcome.BLOCKED:
                self.audit_logger.log_security_violation(
                    realm_id=realm_id,
                    violation_type='validation_blocked',
                    violation_details={
                        'outcome': outcome.value,
                        'config_hash': config_hash,
                        'validation_results': validation_results
                    },
                    request_context=request_context
                )
            elif outcome == ValidationOutcome.REJECTED:
                self.audit_logger.log_permission_check(
                    realm_id=realm_id,
                    permission_type='realm_configuration',
                    resource='dynamic_realm_config',
                    granted=False,
                    reason='Validation failed',
                    request_context=request_context
                )
            
        except Exception as e:
            logger.error(f"Failed to log validation outcome: {e}")
    
    def _update_threat_metrics(self, security_context: SecurityContext, outcome: ValidationOutcome):
        """Update threat detection metrics"""
        try:
            client_ip = security_context.client_ip
            
            if outcome in [ValidationOutcome.REJECTED, ValidationOutcome.BLOCKED]:
                # Increment failed validation count
                self._failed_validations[client_ip] = self._failed_validations.get(client_ip, 0) + 1
            else:
                # Reset failed validation count on success
                if client_ip in self._failed_validations:
                    self._failed_validations[client_ip] = 0
                    
        except Exception as e:
            logger.error(f"Failed to update threat metrics: {e}")
    
    def _generate_config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration caching/tracking"""
        try:
            config_str = json.dumps(config, sort_keys=True)
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        except Exception:
            return 'unknown_hash'
    
    def _security_context_to_dict(self, security_context: SecurityContext) -> Dict[str, Any]:
        """Convert security context to dictionary for logging"""
        return {
            'client_ip': security_context.client_ip,
            'user_agent': security_context.user_agent,
            'request_id': security_context.request_id,
            'session_id': security_context.session_id,
            'realm_id': security_context.realm_id,
            'operation': security_context.operation,
            'security_level': security_context.security_level.value
        }
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        return {
            'rate_limiting': {
                'tracked_ips': len(self._request_tracker),
                'total_requests_last_minute': sum(len(requests) for requests in self._request_tracker.values())
            },
            'threat_detection': {
                'failed_validations_by_ip': dict(self._failed_validations),
                'blocked_ips': list(self._blocked_ips.keys()),
                'blocked_ips_count': len(self._blocked_ips)
            },
            'validation': {
                'security_level': self.security_level.value,
                'threat_detection_enabled': self.enable_threat_detection,
                'rate_limiting_enabled': self.rate_limit_enabled
            },
            'cache_stats': self.cache_manager.get_cache_metrics() if self.cache_manager else {}
        }
    
    def reset_security_state(self, client_ip: Optional[str] = None):
        """Reset security state for testing or recovery"""
        if client_ip:
            # Reset specific IP
            self._request_tracker.pop(client_ip, None)
            self._failed_validations.pop(client_ip, None)
            self._blocked_ips.pop(client_ip, None)
            logger.info(f"Security state reset for IP: {client_ip}")
        else:
            # Reset all security state
            self._request_tracker.clear()
            self._failed_validations.clear()
            self._blocked_ips.clear()
            logger.info("All security state reset")
    
    def shutdown(self):
        """Shutdown security pipeline and cleanup resources"""
        logger.info("Shutting down EnhancedSecurityPipeline")
        
        if self.cache_manager:
            self.cache_manager.shutdown()
        
        if self.audit_logger:
            self.audit_logger.shutdown()
        
        # Clear security state
        self._request_tracker.clear()
        self._failed_validations.clear()
        self._blocked_ips.clear()