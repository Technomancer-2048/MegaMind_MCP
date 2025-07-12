#!/usr/bin/env python3
"""
Realm Security Validator for MegaMind Context Database
Validates realm isolation, access control, and security boundaries
"""

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set, Tuple
from enum import Enum

import mysql.connector

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ViolationType(Enum):
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    REALM_ISOLATION_BREACH = "realm_isolation_breach"
    PERMISSION_ESCALATION = "permission_escalation"
    DATA_LEAKAGE = "data_leakage"
    INJECTION_ATTEMPT = "injection_attempt"
    SUSPICIOUS_PATTERN = "suspicious_pattern"

@dataclass
class SecurityViolation:
    """Represents a security violation with detailed context"""
    violation_id: str
    violation_type: ViolationType
    severity: SecurityLevel
    user_id: Optional[str]
    source_ip: str
    attempted_action: str
    target_resource: str
    realm_context: str
    detected_at: datetime
    details: Dict[str, Any]

@dataclass
class RealmIsolationCheck:
    """Results of realm isolation validation"""
    is_isolated: bool
    violations: List[str]
    affected_chunks: List[str]
    cross_realm_leaks: List[Dict[str, Any]]
    security_score: float  # 0.0 to 1.0

@dataclass
class AccessControlValidation:
    """Results of access control validation"""
    is_valid: bool
    permission_violations: List[str]
    role_conflicts: List[str]
    escalation_risks: List[str]
    user_access_matrix: Dict[str, List[str]]

class RealmSecurityValidator:
    """Validates security boundaries and access control in realm system"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logging.getLogger(__name__)
        
        # Security thresholds
        self.max_failed_attempts = 5
        self.suspicious_activity_window = timedelta(minutes=15)
        self.isolation_check_interval = timedelta(hours=24)
        
    # ===================================================================
    # Realm Isolation Validation
    # ===================================================================
    
    def validate_realm_isolation(self, realm_id: str) -> RealmIsolationCheck:
        """Comprehensive realm isolation validation"""
        try:
            violations = []
            affected_chunks = []
            cross_realm_leaks = []
            
            # Check for unauthorized cross-realm data access
            unauthorized_access = self._check_unauthorized_cross_realm_access(realm_id)
            if unauthorized_access:
                violations.extend(unauthorized_access)
            
            # Check for data leakage through relationships
            relationship_leaks = self._check_relationship_leakage(realm_id)
            if relationship_leaks:
                cross_realm_leaks.extend(relationship_leaks)
                violations.append(f"Found {len(relationship_leaks)} potential relationship-based data leaks")
            
            # Check for session context violations
            session_violations = self._check_session_context_violations(realm_id)
            if session_violations:
                violations.extend(session_violations)
            
            # Check for inheritance bypass attempts
            inheritance_bypasses = self._check_inheritance_bypass_attempts(realm_id)
            if inheritance_bypasses:
                violations.extend(inheritance_bypasses)
            
            # Check for permission escalation through realm switching
            escalation_attempts = self._check_realm_escalation_attempts(realm_id)
            if escalation_attempts:
                violations.extend(escalation_attempts)
            
            # Calculate security score based on violations
            total_checks = 5
            failed_checks = len([v for v in [unauthorized_access, relationship_leaks, 
                               session_violations, inheritance_bypasses, escalation_attempts] if v])
            security_score = max(0.0, 1.0 - (failed_checks / total_checks))
            
            is_isolated = len(violations) == 0
            
            result = RealmIsolationCheck(
                is_isolated=is_isolated,
                violations=violations,
                affected_chunks=affected_chunks,
                cross_realm_leaks=cross_realm_leaks,
                security_score=security_score
            )
            
            # Log security assessment
            self.logger.info(f"Realm isolation check for {realm_id}: "
                           f"isolated={is_isolated}, score={security_score:.2f}, "
                           f"violations={len(violations)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to validate realm isolation for {realm_id}: {e}")
            return RealmIsolationCheck(
                is_isolated=False,
                violations=[f"Validation error: {str(e)}"],
                affected_chunks=[],
                cross_realm_leaks=[],
                security_score=0.0
            )
    
    def _check_unauthorized_cross_realm_access(self, realm_id: str) -> List[str]:
        """Check for unauthorized cross-realm data access"""
        violations = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for chunks accessed by users without proper realm permissions
            query = """
            SELECT DISTINCT c.chunk_id, c.realm_id, al.user_id, al.event_timestamp
            FROM megamind_chunks c
            JOIN megamind_audit_log al ON al.target_id = c.chunk_id AND al.target_type = 'chunk'
            WHERE c.realm_id = %s 
              AND al.event_type = 'chunk_accessed'
              AND al.event_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
              AND NOT EXISTS (
                  SELECT 1 FROM megamind_user_role_assignments ura
                  JOIN megamind_realm_roles rr ON ura.role_id = rr.role_id
                  WHERE ura.user_id = al.user_id 
                    AND (ura.realm_id = c.realm_id OR ura.realm_id IS NULL)
                    AND ura.is_active = TRUE
                    AND rr.can_read = TRUE
              )
            """
            
            cursor.execute(query, (realm_id,))
            unauthorized_accesses = cursor.fetchall()
            
            for access in unauthorized_accesses:
                violations.append(
                    f"Unauthorized access to chunk {access['chunk_id']} by user {access['user_id']}"
                )
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check unauthorized cross-realm access: {e}")
            return [f"Access check error: {str(e)}"]
        finally:
            if cursor:
                cursor.close()
    
    def _check_relationship_leakage(self, realm_id: str) -> List[Dict[str, Any]]:
        """Check for data leakage through cross-realm relationships"""
        leaks = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Find relationships that might leak data across realm boundaries
            query = """
            SELECT cr.relationship_id, cr.chunk_id, cr.related_chunk_id,
                   c1.realm_id AS source_realm, c2.realm_id AS target_realm,
                   cr.relationship_type, cr.strength
            FROM megamind_chunk_relationships cr
            JOIN megamind_chunks c1 ON cr.chunk_id = c1.chunk_id
            JOIN megamind_chunks c2 ON cr.related_chunk_id = c2.chunk_id
            WHERE (c1.realm_id = %s OR c2.realm_id = %s)
              AND c1.realm_id != c2.realm_id
              AND NOT EXISTS (
                  SELECT 1 FROM megamind_realm_inheritance ri
                  WHERE (ri.child_realm_id = c1.realm_id AND ri.parent_realm_id = c2.realm_id)
                     OR (ri.child_realm_id = c2.realm_id AND ri.parent_realm_id = c1.realm_id)
              )
            """
            
            cursor.execute(query, (realm_id, realm_id))
            potential_leaks = cursor.fetchall()
            
            for leak in potential_leaks:
                leaks.append({
                    'relationship_id': leak['relationship_id'],
                    'source_chunk': leak['chunk_id'],
                    'target_chunk': leak['related_chunk_id'],
                    'source_realm': leak['source_realm'],
                    'target_realm': leak['target_realm'],
                    'relationship_type': leak['relationship_type'],
                    'strength': float(leak['strength']),
                    'risk_level': 'high' if leak['strength'] > 0.8 else 'medium'
                })
            
            return leaks
            
        except Exception as e:
            self.logger.error(f"Failed to check relationship leakage: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def _check_session_context_violations(self, realm_id: str) -> List[str]:
        """Check for session context violations"""
        violations = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for sessions accessing data outside their configured realm
            query = """
            SELECT DISTINCT sm.session_id, sm.realm_id AS session_realm, 
                   al.target_realm_id AS accessed_realm, al.user_id
            FROM megamind_session_metadata sm
            JOIN megamind_audit_log al ON sm.session_id = al.session_id
            WHERE sm.realm_id = %s
              AND al.target_realm_id IS NOT NULL
              AND al.target_realm_id != sm.realm_id
              AND al.event_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
              AND NOT EXISTS (
                  SELECT 1 FROM megamind_realm_inheritance ri
                  WHERE ri.child_realm_id = sm.realm_id 
                    AND ri.parent_realm_id = al.target_realm_id
              )
            """
            
            cursor.execute(query, (realm_id,))
            context_violations = cursor.fetchall()
            
            for violation in context_violations:
                violations.append(
                    f"Session {violation['session_id']} accessed realm {violation['accessed_realm']} "
                    f"outside its context (configured for {violation['session_realm']})"
                )
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check session context violations: {e}")
            return [f"Session context check error: {str(e)}"]
        finally:
            if cursor:
                cursor.close()
    
    def _check_inheritance_bypass_attempts(self, realm_id: str) -> List[str]:
        """Check for attempts to bypass inheritance controls"""
        violations = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for direct access attempts to parent realm data when selective inheritance is configured
            query = """
            SELECT ri.child_realm_id, ri.parent_realm_id, ri.inheritance_config,
                   COUNT(al.audit_id) AS access_attempts
            FROM megamind_realm_inheritance ri
            JOIN megamind_audit_log al ON al.target_realm_id = ri.parent_realm_id
            WHERE ri.child_realm_id = %s
              AND ri.inheritance_type = 'selective'
              AND al.event_type = 'chunk_accessed'
              AND al.event_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
            GROUP BY ri.child_realm_id, ri.parent_realm_id, ri.inheritance_config
            HAVING access_attempts > 10  -- Threshold for suspicious activity
            """
            
            cursor.execute(query, (realm_id,))
            bypass_attempts = cursor.fetchall()
            
            for attempt in bypass_attempts:
                violations.append(
                    f"Potential inheritance bypass: {attempt['access_attempts']} attempts to access "
                    f"parent realm {attempt['parent_realm_id']} with selective inheritance configured"
                )
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check inheritance bypass attempts: {e}")
            return [f"Inheritance bypass check error: {str(e)}"]
        finally:
            if cursor:
                cursor.close()
    
    def _check_realm_escalation_attempts(self, realm_id: str) -> List[str]:
        """Check for permission escalation through realm switching"""
        violations = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for users attempting to access higher-privilege realms
            query = """
            SELECT al.user_id, al.target_realm_id, COUNT(*) AS attempt_count,
                   MIN(al.event_timestamp) AS first_attempt,
                   MAX(al.event_timestamp) AS last_attempt
            FROM megamind_audit_log al
            WHERE al.event_type = 'permission_denied'
              AND al.target_realm_id = %s
              AND al.event_timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
            GROUP BY al.user_id, al.target_realm_id
            HAVING attempt_count >= 3  -- Threshold for suspicious activity
            """
            
            cursor.execute(query, (realm_id,))
            escalation_attempts = cursor.fetchall()
            
            for attempt in escalation_attempts:
                violations.append(
                    f"Potential escalation attempt: User {attempt['user_id']} made "
                    f"{attempt['attempt_count']} denied access attempts to realm {realm_id}"
                )
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check realm escalation attempts: {e}")
            return [f"Escalation check error: {str(e)}"]
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Access Control Validation
    # ===================================================================
    
    def validate_access_control(self, realm_id: str) -> AccessControlValidation:
        """Validate access control configuration and enforcement"""
        try:
            permission_violations = []
            role_conflicts = []
            escalation_risks = []
            user_access_matrix = {}
            
            # Check for permission violations
            permission_violations = self._check_permission_violations(realm_id)
            
            # Check for role conflicts
            role_conflicts = self._check_role_conflicts(realm_id)
            
            # Check for escalation risks
            escalation_risks = self._check_escalation_risks(realm_id)
            
            # Generate user access matrix
            user_access_matrix = self._generate_user_access_matrix(realm_id)
            
            is_valid = (len(permission_violations) == 0 and 
                       len(role_conflicts) == 0 and 
                       len(escalation_risks) == 0)
            
            result = AccessControlValidation(
                is_valid=is_valid,
                permission_violations=permission_violations,
                role_conflicts=role_conflicts,
                escalation_risks=escalation_risks,
                user_access_matrix=user_access_matrix
            )
            
            self.logger.info(f"Access control validation for {realm_id}: "
                           f"valid={is_valid}, violations={len(permission_violations)}, "
                           f"conflicts={len(role_conflicts)}, risks={len(escalation_risks)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to validate access control for {realm_id}: {e}")
            return AccessControlValidation(
                is_valid=False,
                permission_violations=[f"Validation error: {str(e)}"],
                role_conflicts=[],
                escalation_risks=[],
                user_access_matrix={}
            )
    
    def _check_permission_violations(self, realm_id: str) -> List[str]:
        """Check for permission violations"""
        violations = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for users with conflicting permissions
            query = """
            SELECT ura.user_id, 
                   GROUP_CONCAT(DISTINCT rr.role_name) AS roles,
                   SUM(rr.can_write) AS write_permissions,
                   SUM(rr.can_delete) AS delete_permissions,
                   SUM(rr.can_manage_realm) AS admin_permissions
            FROM megamind_user_role_assignments ura
            JOIN megamind_realm_roles rr ON ura.role_id = rr.role_id
            WHERE (ura.realm_id = %s OR ura.realm_id IS NULL)
              AND ura.is_active = TRUE
              AND (ura.expires_at IS NULL OR ura.expires_at > NOW())
            GROUP BY ura.user_id
            HAVING admin_permissions > 1 OR (write_permissions > 0 AND delete_permissions > 0)
            """
            
            cursor.execute(query, (realm_id,))
            permission_conflicts = cursor.fetchall()
            
            for conflict in permission_conflicts:
                violations.append(
                    f"User {conflict['user_id']} has conflicting permissions: "
                    f"roles={conflict['roles']}, write={conflict['write_permissions']}, "
                    f"delete={conflict['delete_permissions']}, admin={conflict['admin_permissions']}"
                )
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check permission violations: {e}")
            return [f"Permission check error: {str(e)}"]
        finally:
            if cursor:
                cursor.close()
    
    def _check_role_conflicts(self, realm_id: str) -> List[str]:
        """Check for role conflicts and overlaps"""
        conflicts = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for users with multiple admin roles
            query = """
            SELECT ura.user_id, COUNT(*) AS admin_role_count,
                   GROUP_CONCAT(rr.role_name) AS admin_roles
            FROM megamind_user_role_assignments ura
            JOIN megamind_realm_roles rr ON ura.role_id = rr.role_id
            WHERE (ura.realm_id = %s OR ura.realm_id IS NULL)
              AND ura.is_active = TRUE
              AND rr.can_manage_realm = TRUE
            GROUP BY ura.user_id
            HAVING admin_role_count > 1
            """
            
            cursor.execute(query, (realm_id,))
            admin_conflicts = cursor.fetchall()
            
            for conflict in admin_conflicts:
                conflicts.append(
                    f"User {conflict['user_id']} has multiple admin roles: {conflict['admin_roles']}"
                )
            
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Failed to check role conflicts: {e}")
            return [f"Role conflict check error: {str(e)}"]
        finally:
            if cursor:
                cursor.close()
    
    def _check_escalation_risks(self, realm_id: str) -> List[str]:
        """Check for privilege escalation risks"""
        risks = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for temporary elevated permissions that might be abused
            query = """
            SELECT ura.user_id, ura.role_id, rr.role_name, ura.expires_at,
                   TIMESTAMPDIFF(HOUR, NOW(), ura.expires_at) AS hours_remaining
            FROM megamind_user_role_assignments ura
            JOIN megamind_realm_roles rr ON ura.role_id = rr.role_id
            WHERE (ura.realm_id = %s OR ura.realm_id IS NULL)
              AND ura.is_active = TRUE
              AND ura.expires_at IS NOT NULL
              AND ura.expires_at > NOW()
              AND (rr.can_manage_realm = TRUE OR rr.can_approve_promotions = TRUE)
            ORDER BY ura.expires_at ASC
            """
            
            cursor.execute(query, (realm_id,))
            elevated_permissions = cursor.fetchall()
            
            for perm in elevated_permissions:
                if perm['hours_remaining'] > 168:  # More than 1 week
                    risks.append(
                        f"Long-term elevated permission: User {perm['user_id']} has "
                        f"{perm['role_name']} role for {perm['hours_remaining']} more hours"
                    )
            
            return risks
            
        except Exception as e:
            self.logger.error(f"Failed to check escalation risks: {e}")
            return [f"Escalation risk check error: {str(e)}"]
        finally:
            if cursor:
                cursor.close()
    
    def _generate_user_access_matrix(self, realm_id: str) -> Dict[str, List[str]]:
        """Generate user access matrix for the realm"""
        matrix = {}
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            query = """
            SELECT ura.user_id, 
                   GROUP_CONCAT(DISTINCT rr.role_name) AS roles,
                   MAX(rr.can_read) AS can_read,
                   MAX(rr.can_write) AS can_write,
                   MAX(rr.can_delete) AS can_delete,
                   MAX(rr.can_promote) AS can_promote,
                   MAX(rr.can_approve_promotions) AS can_approve,
                   MAX(rr.can_manage_users) AS can_manage_users,
                   MAX(rr.can_manage_realm) AS can_manage_realm
            FROM megamind_user_role_assignments ura
            JOIN megamind_realm_roles rr ON ura.role_id = rr.role_id
            WHERE (ura.realm_id = %s OR ura.realm_id IS NULL)
              AND ura.is_active = TRUE
              AND (ura.expires_at IS NULL OR ura.expires_at > NOW())
            GROUP BY ura.user_id
            """
            
            cursor.execute(query, (realm_id,))
            user_permissions = cursor.fetchall()
            
            for user in user_permissions:
                permissions = []
                if user['can_read']: permissions.append('read')
                if user['can_write']: permissions.append('write')
                if user['can_delete']: permissions.append('delete')
                if user['can_promote']: permissions.append('promote')
                if user['can_approve']: permissions.append('approve_promotions')
                if user['can_manage_users']: permissions.append('manage_users')
                if user['can_manage_realm']: permissions.append('manage_realm')
                
                matrix[user['user_id']] = permissions
            
            return matrix
            
        except Exception as e:
            self.logger.error(f"Failed to generate user access matrix: {e}")
            return {}
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Security Monitoring and Alerting
    # ===================================================================
    
    def detect_suspicious_activity(self, user_id: str, source_ip: str) -> List[SecurityViolation]:
        """Detect suspicious activity patterns"""
        violations = []
        
        try:
            # Check for rapid successive failed attempts
            failed_attempts = self._check_failed_attempts(user_id, source_ip)
            if failed_attempts:
                violations.extend(failed_attempts)
            
            # Check for unusual access patterns
            unusual_patterns = self._check_unusual_access_patterns(user_id)
            if unusual_patterns:
                violations.extend(unusual_patterns)
            
            # Check for privilege escalation attempts
            escalation_attempts = self._check_privilege_escalation_attempts(user_id)
            if escalation_attempts:
                violations.extend(escalation_attempts)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to detect suspicious activity: {e}")
            return []
    
    def _check_failed_attempts(self, user_id: str, source_ip: str) -> List[SecurityViolation]:
        """Check for excessive failed attempts"""
        violations = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Count failed attempts in the last 15 minutes
            query = """
            SELECT COUNT(*) AS failed_count
            FROM megamind_audit_log
            WHERE (user_id = %s OR source_ip = %s)
              AND event_type = 'permission_denied'
              AND event_timestamp >= DATE_SUB(NOW(), INTERVAL 15 MINUTE)
            """
            
            cursor.execute(query, (user_id, source_ip))
            result = cursor.fetchone()
            
            if result and result['failed_count'] >= self.max_failed_attempts:
                violation = SecurityViolation(
                    violation_id=f"violation_{uuid.uuid4().hex[:12]}",
                    violation_type=ViolationType.SUSPICIOUS_PATTERN,
                    severity=SecurityLevel.MEDIUM,
                    user_id=user_id,
                    source_ip=source_ip,
                    attempted_action="Multiple failed access attempts",
                    target_resource="system",
                    realm_context="multiple",
                    detected_at=datetime.now(),
                    details={'failed_count': result['failed_count']}
                )
                violations.append(violation)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check failed attempts: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def _check_unusual_access_patterns(self, user_id: str) -> List[SecurityViolation]:
        """Check for unusual access patterns"""
        violations = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for access outside normal hours (simplified check)
            query = """
            SELECT COUNT(*) AS off_hours_count
            FROM megamind_audit_log
            WHERE user_id = %s
              AND event_type = 'chunk_accessed'
              AND event_timestamp >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
              AND (HOUR(event_timestamp) < 6 OR HOUR(event_timestamp) > 22)
            """
            
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            
            if result and result['off_hours_count'] >= 10:  # Threshold for suspicious activity
                violation = SecurityViolation(
                    violation_id=f"violation_{uuid.uuid4().hex[:12]}",
                    violation_type=ViolationType.SUSPICIOUS_PATTERN,
                    severity=SecurityLevel.LOW,
                    user_id=user_id,
                    source_ip="unknown",
                    attempted_action="Unusual access timing",
                    target_resource="chunks",
                    realm_context="multiple",
                    detected_at=datetime.now(),
                    details={'off_hours_access_count': result['off_hours_count']}
                )
                violations.append(violation)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check unusual access patterns: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def _check_privilege_escalation_attempts(self, user_id: str) -> List[SecurityViolation]:
        """Check for privilege escalation attempts"""
        violations = []
        
        try:
            cursor = self.db.cursor(dictionary=True)
            
            # Check for attempts to access admin functions without permissions
            query = """
            SELECT COUNT(*) AS escalation_attempts,
                   GROUP_CONCAT(DISTINCT target_type) AS attempted_targets
            FROM megamind_audit_log
            WHERE user_id = %s
              AND event_type = 'permission_denied'
              AND event_timestamp >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
              AND (event_description LIKE '%admin%' OR event_description LIKE '%manage%')
            """
            
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            
            if result and result['escalation_attempts'] >= 3:
                violation = SecurityViolation(
                    violation_id=f"violation_{uuid.uuid4().hex[:12]}",
                    violation_type=ViolationType.PERMISSION_ESCALATION,
                    severity=SecurityLevel.HIGH,
                    user_id=user_id,
                    source_ip="unknown",
                    attempted_action="Privilege escalation attempt",
                    target_resource=result['attempted_targets'] or "unknown",
                    realm_context="multiple",
                    detected_at=datetime.now(),
                    details={'escalation_attempts': result['escalation_attempts']}
                )
                violations.append(violation)
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Failed to check privilege escalation attempts: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def run_comprehensive_security_scan(self, realm_id: str) -> Dict[str, Any]:
        """Run comprehensive security scan for a realm"""
        try:
            self.logger.info(f"Starting comprehensive security scan for realm {realm_id}")
            
            # Validate realm isolation
            isolation_check = self.validate_realm_isolation(realm_id)
            
            # Validate access control
            access_control_check = self.validate_access_control(realm_id)
            
            # Generate security summary
            total_violations = (len(isolation_check.violations) + 
                              len(access_control_check.permission_violations) + 
                              len(access_control_check.role_conflicts) + 
                              len(access_control_check.escalation_risks))
            
            overall_security_score = (isolation_check.security_score + 
                                    (1.0 if access_control_check.is_valid else 0.0)) / 2.0
            
            security_level = SecurityLevel.CRITICAL if total_violations > 10 else \
                           SecurityLevel.HIGH if total_violations > 5 else \
                           SecurityLevel.MEDIUM if total_violations > 2 else \
                           SecurityLevel.LOW
            
            scan_result = {
                'realm_id': realm_id,
                'scan_timestamp': datetime.now().isoformat(),
                'overall_security_score': overall_security_score,
                'security_level': security_level.value,
                'total_violations': total_violations,
                'isolation_check': {
                    'is_isolated': isolation_check.is_isolated,
                    'security_score': isolation_check.security_score,
                    'violations': isolation_check.violations,
                    'cross_realm_leaks': isolation_check.cross_realm_leaks
                },
                'access_control_check': {
                    'is_valid': access_control_check.is_valid,
                    'permission_violations': access_control_check.permission_violations,
                    'role_conflicts': access_control_check.role_conflicts,
                    'escalation_risks': access_control_check.escalation_risks,
                    'user_count': len(access_control_check.user_access_matrix)
                },
                'recommendations': self._generate_security_recommendations(
                    isolation_check, access_control_check
                )
            }
            
            self.logger.info(f"Security scan completed for realm {realm_id}: "
                           f"score={overall_security_score:.2f}, level={security_level.value}")
            
            return scan_result
            
        except Exception as e:
            self.logger.error(f"Failed to run comprehensive security scan: {e}")
            return {
                'realm_id': realm_id,
                'scan_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall_security_score': 0.0,
                'security_level': SecurityLevel.CRITICAL.value
            }
    
    def _generate_security_recommendations(self, isolation_check: RealmIsolationCheck,
                                         access_control_check: AccessControlValidation) -> List[str]:
        """Generate security recommendations based on scan results"""
        recommendations = []
        
        if not isolation_check.is_isolated:
            recommendations.append("Review and strengthen realm isolation controls")
            if isolation_check.cross_realm_leaks:
                recommendations.append("Audit cross-realm relationships for potential data leakage")
        
        if not access_control_check.is_valid:
            recommendations.append("Review role assignments and permission configurations")
            if access_control_check.role_conflicts:
                recommendations.append("Resolve role conflicts to prevent permission overlap")
            if access_control_check.escalation_risks:
                recommendations.append("Review temporary elevated permissions and consider reducing duration")
        
        if isolation_check.security_score < 0.7:
            recommendations.append("Implement additional monitoring for suspicious activity")
        
        if len(access_control_check.user_access_matrix) > 50:
            recommendations.append("Consider implementing role hierarchy to simplify permission management")
        
        return recommendations