#!/usr/bin/env python3
"""
Dynamic Realm Configuration Validator
Provides security validation, sanitization, and audit logging for dynamic realm configurations
"""

import re
import json
import logging
import hashlib
from typing import Dict, Any, Tuple, Set, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of validation check"""
    valid: bool
    severity: ValidationSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    
class RealmConfigValidator:
    """Validates dynamic realm configurations for security and correctness"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Security constraints
        self.allowed_realms = set(self.config.get('allowed_realms', ['PROJECT', 'GLOBAL']))
        self.max_realm_name_length = self.config.get('max_realm_name_length', 100)
        self.max_project_name_length = self.config.get('max_project_name_length', 200)
        
        # Restricted patterns for realm IDs
        self.restricted_patterns = {
            'admin', 'root', 'system', 'internal', 'test', 'debug', 
            'config', 'secret', 'private', 'auth', 'security'
        }
        
        # Allowed characters in realm IDs (alphanumeric, underscore, hyphen)
        self.realm_id_pattern = re.compile(r'^[A-Za-z0-9_-]+$')
        
        # Cache for validation results to improve performance
        self._validation_cache: Dict[str, Tuple[ValidationResult, datetime]] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        logger.info("RealmConfigValidator initialized with security constraints")
    
    def validate_realm_config(self, config: Dict[str, Any], skip_cache: bool = False) -> List[ValidationResult]:
        """
        Comprehensive validation of dynamic realm configuration
        Returns list of validation results (errors, warnings, info)
        """
        try:
            # Generate cache key from config hash
            config_hash = self._hash_config(config)
            
            # Check cache if not skipping
            if not skip_cache and config_hash in self._validation_cache:
                cached_result, timestamp = self._validation_cache[config_hash]
                if datetime.now() - timestamp < self._cache_ttl:
                    logger.debug(f"Using cached validation result for config hash: {config_hash[:8]}")
                    return [cached_result]
            
            results = []
            
            # 1. Required fields validation
            results.extend(self._validate_required_fields(config))
            
            # 2. Realm ID validation
            results.extend(self._validate_realm_id(config.get('project_realm', '')))
            
            # 3. Project name validation
            results.extend(self._validate_project_name(config.get('project_name', '')))
            
            # 4. Default target validation
            results.extend(self._validate_default_target(config.get('default_target', '')))
            
            # 5. Numeric parameters validation
            results.extend(self._validate_numeric_parameters(config))
            
            # 6. Security constraints validation
            results.extend(self._validate_security_constraints(config))
            
            # 7. Cross-field validation
            results.extend(self._validate_cross_field_constraints(config))
            
            # Cache the overall result
            overall_valid = not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for r in results)
            overall_result = ValidationResult(
                valid=overall_valid,
                severity=ValidationSeverity.INFO if overall_valid else ValidationSeverity.ERROR,
                message=f"Configuration validation {'passed' if overall_valid else 'failed'} with {len(results)} checks",
                details={'validation_count': len(results), 'config_hash': config_hash}
            )
            
            if not skip_cache:
                self._validation_cache[config_hash] = (overall_result, datetime.now())
            
            results.insert(0, overall_result)
            return results
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            return [ValidationResult(
                valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Validation exception: {str(e)}"
            )]
    
    def _validate_required_fields(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate that all required fields are present"""
        results = []
        required_fields = ['project_realm', 'project_name', 'default_target']
        
        missing_fields = [field for field in required_fields if field not in config or not config[field]]
        
        if missing_fields:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Missing required fields: {missing_fields}",
                details={'missing_fields': missing_fields, 'required_fields': required_fields}
            ))
        else:
            results.append(ValidationResult(
                valid=True,
                severity=ValidationSeverity.INFO,
                message="All required fields present"
            ))
        
        return results
    
    def _validate_realm_id(self, realm_id: str) -> List[ValidationResult]:
        """Validate realm ID format and security constraints"""
        results = []
        
        if not realm_id:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message="Realm ID cannot be empty"
            ))
            return results
        
        # Length check
        if len(realm_id) > self.max_realm_name_length:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Realm ID too long: {len(realm_id)} > {self.max_realm_name_length}",
                details={'realm_id_length': len(realm_id), 'max_length': self.max_realm_name_length}
            ))
        
        # Character format check
        if not self.realm_id_pattern.match(realm_id):
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid realm ID format: {realm_id}. Only alphanumeric, underscore, and hyphen allowed",
                details={'realm_id': realm_id, 'pattern': self.realm_id_pattern.pattern}
            ))
        
        # Restricted patterns check
        realm_lower = realm_id.lower()
        matching_patterns = [pattern for pattern in self.restricted_patterns if pattern in realm_lower]
        if matching_patterns:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Realm ID contains restricted patterns: {matching_patterns}",
                details={'realm_id': realm_id, 'matching_patterns': matching_patterns}
            ))
        
        # Reserved names check
        if realm_lower in {'null', 'undefined', 'none', 'empty', 'default'}:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Realm ID uses reserved name: {realm_id}"
            ))
        
        if not results:
            results.append(ValidationResult(
                valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Realm ID validation passed: {realm_id}"
            ))
        
        return results
    
    def _validate_project_name(self, project_name: str) -> List[ValidationResult]:
        """Validate project name"""
        results = []
        
        if not project_name:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message="Project name cannot be empty"
            ))
            return results
        
        # Length check
        if len(project_name) > self.max_project_name_length:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Project name too long: {len(project_name)} > {self.max_project_name_length}"
            ))
        
        # Character safety check (prevent XSS, injection)
        dangerous_chars = ['<', '>', '"', "'", '&', '\\', '/', '\x00']
        found_dangerous = [char for char in dangerous_chars if char in project_name]
        if found_dangerous:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Project name contains potentially dangerous characters: {found_dangerous}",
                details={'dangerous_chars': found_dangerous}
            ))
        
        if not results:
            results.append(ValidationResult(
                valid=True,
                severity=ValidationSeverity.INFO,
                message="Project name validation passed"
            ))
        
        return results
    
    def _validate_default_target(self, default_target: str) -> List[ValidationResult]:
        """Validate default target value"""
        results = []
        
        valid_targets = ['PROJECT', 'GLOBAL']
        
        if default_target not in valid_targets:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid default_target: {default_target}. Must be one of: {valid_targets}",
                details={'provided_target': default_target, 'valid_targets': valid_targets}
            ))
        else:
            results.append(ValidationResult(
                valid=True,
                severity=ValidationSeverity.INFO,
                message=f"Default target validation passed: {default_target}"
            ))
        
        return results
    
    def _validate_numeric_parameters(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate numeric configuration parameters"""
        results = []
        
        numeric_params = {
            'project_priority_weight': (0.1, 5.0, 1.2),
            'global_priority_weight': (0.1, 5.0, 1.0)
        }
        
        for param_name, (min_val, max_val, default_val) in numeric_params.items():
            if param_name in config:
                try:
                    value = float(config[param_name])
                    if not (min_val <= value <= max_val):
                        results.append(ValidationResult(
                            valid=False,
                            severity=ValidationSeverity.WARNING,
                            message=f"{param_name} out of range: {value} not in [{min_val}, {max_val}]",
                            details={'parameter': param_name, 'value': value, 'range': [min_val, max_val]}
                        ))
                    else:
                        results.append(ValidationResult(
                            valid=True,
                            severity=ValidationSeverity.INFO,
                            message=f"{param_name} validation passed: {value}"
                        ))
                except (ValueError, TypeError):
                    results.append(ValidationResult(
                        valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"{param_name} is not a valid number: {config[param_name]}"
                    ))
        
        return results
    
    def _validate_security_constraints(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate security-specific constraints"""
        results = []
        
        # Check for potential injection attempts
        dangerous_strings = ['script', 'eval', 'exec', 'import', 'os.', 'sys.', '__']
        config_str = json.dumps(config).lower()
        
        found_dangerous = [s for s in dangerous_strings if s in config_str]
        if found_dangerous:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Configuration contains potentially dangerous strings: {found_dangerous}",
                details={'dangerous_strings': found_dangerous}
            ))
        
        # Check for excessive nesting or size
        config_size = len(json.dumps(config))
        if config_size > 10000:  # 10KB limit
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Configuration too large: {config_size} bytes > 10KB limit"
            ))
        
        if not found_dangerous and config_size <= 10000:
            results.append(ValidationResult(
                valid=True,
                severity=ValidationSeverity.INFO,
                message="Security constraints validation passed"
            ))
        
        return results
    
    def _validate_cross_field_constraints(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate constraints that involve multiple fields"""
        results = []
        
        # Ensure project_realm and global_realm are different
        project_realm = config.get('project_realm', '')
        global_realm = config.get('global_realm', 'GLOBAL')
        
        if project_realm == global_realm:
            results.append(ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Project realm cannot be same as global realm: {project_realm}",
                details={'project_realm': project_realm, 'global_realm': global_realm}
            ))
        
        # Validate priority weights relationship
        project_weight = config.get('project_priority_weight', 1.2)
        global_weight = config.get('global_priority_weight', 1.0)
        
        try:
            if float(project_weight) <= 0 or float(global_weight) <= 0:
                results.append(ValidationResult(
                    valid=False,
                    severity=ValidationSeverity.ERROR,
                    message="Priority weights must be positive"
                ))
        except (ValueError, TypeError):
            pass  # Handled in numeric validation
        
        if not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for r in results):
            results.append(ValidationResult(
                valid=True,
                severity=ValidationSeverity.INFO,
                message="Cross-field constraints validation passed"
            ))
        
        return results
    
    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration values for safety"""
        try:
            sanitized = config.copy()
            
            # Sanitize project name - remove dangerous characters
            if 'project_name' in sanitized:
                sanitized['project_name'] = re.sub(r'[<>"\'\&\\\/\x00]', '', sanitized['project_name'])
                sanitized['project_name'] = sanitized['project_name'][:self.max_project_name_length]
            
            # Sanitize realm ID - ensure alphanumeric only
            if 'project_realm' in sanitized:
                sanitized['project_realm'] = re.sub(r'[^A-Za-z0-9_-]', '', sanitized['project_realm'])
                sanitized['project_realm'] = sanitized['project_realm'][:self.max_realm_name_length]
            
            # Clamp numeric values
            if 'project_priority_weight' in sanitized:
                try:
                    sanitized['project_priority_weight'] = max(0.1, min(5.0, float(sanitized['project_priority_weight'])))
                except (ValueError, TypeError):
                    sanitized['project_priority_weight'] = 1.2
            
            if 'global_priority_weight' in sanitized:
                try:
                    sanitized['global_priority_weight'] = max(0.1, min(5.0, float(sanitized['global_priority_weight'])))
                except (ValueError, TypeError):
                    sanitized['global_priority_weight'] = 1.0
            
            # Ensure boolean values
            if 'cross_realm_search_enabled' in sanitized:
                sanitized['cross_realm_search_enabled'] = bool(sanitized['cross_realm_search_enabled'])
            
            # Set defaults for missing optional fields
            sanitized.setdefault('global_realm', 'GLOBAL')
            sanitized.setdefault('cross_realm_search_enabled', True)
            sanitized.setdefault('project_priority_weight', 1.2)
            sanitized.setdefault('global_priority_weight', 1.0)
            
            logger.debug(f"Configuration sanitized successfully")
            return sanitized
            
        except Exception as e:
            logger.error(f"Configuration sanitization failed: {e}")
            # Return safe defaults
            return {
                'project_realm': 'PROJECT',
                'project_name': 'Default Project',
                'default_target': 'PROJECT',
                'global_realm': 'GLOBAL',
                'cross_realm_search_enabled': True,
                'project_priority_weight': 1.2,
                'global_priority_weight': 1.0
            }
    
    def is_valid_configuration(self, config: Dict[str, Any]) -> bool:
        """Quick validation check - returns True if configuration is valid"""
        results = self.validate_realm_config(config)
        return not any(r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for r in results)
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate summary of validation results"""
        summary = {
            'total_checks': len(results),
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        for result in results:
            if result.severity == ValidationSeverity.CRITICAL or result.severity == ValidationSeverity.ERROR:
                summary['valid'] = False
                summary['errors'].append(result.message)
            elif result.severity == ValidationSeverity.WARNING:
                summary['warnings'].append(result.message)
            else:
                summary['info'].append(result.message)
        
        return summary
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash for configuration caching"""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear validation cache"""
        self._validation_cache.clear()
        logger.debug("Validation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        now = datetime.now()
        valid_entries = sum(1 for _, timestamp in self._validation_cache.values() if now - timestamp < self._cache_ttl)
        
        return {
            'total_entries': len(self._validation_cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self._validation_cache) - valid_entries,
            'cache_ttl_minutes': self._cache_ttl.total_seconds() / 60
        }