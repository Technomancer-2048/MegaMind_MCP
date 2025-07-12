#!/usr/bin/env python3
"""
Knowledge Promotion Manager for MegaMind Context Database
Handles promotion workflows, approval processes, and role-based access control
"""

import json
import logging
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from enum import Enum

import mysql.connector

logger = logging.getLogger(__name__)

class PromotionType(Enum):
    COPY = "copy"
    MOVE = "move"
    REFERENCE = "reference"

class PromotionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    PROCESSING = "processing"
    COMPLETED = "completed"

class BusinessImpact(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PermissionType(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    PROMOTE = "promote"
    APPROVE_PROMOTIONS = "approve_promotions"
    MANAGE_USERS = "manage_users"
    MANAGE_REALM = "manage_realm"

@dataclass
class PromotionRequest:
    """Represents a knowledge promotion request"""
    promotion_id: str
    source_chunk_id: str
    source_realm_id: str
    target_realm_id: str
    promotion_type: PromotionType
    status: PromotionStatus
    requested_by: str
    requested_at: datetime
    justification: str
    business_impact: BusinessImpact
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    target_chunk_id: Optional[str] = None
    original_content: Optional[str] = None

@dataclass
class PromotionImpact:
    """Represents impact analysis for a promotion"""
    impact_id: str
    promotion_id: str
    affected_chunks_count: int
    affected_relationships_count: int
    potential_conflicts_count: int
    content_quality_score: float
    relevance_score: float
    uniqueness_score: float
    conflict_analysis: Optional[Dict] = None
    dependency_analysis: Optional[Dict] = None
    usage_impact: Optional[Dict] = None

@dataclass
class UserRole:
    """Represents a user role assignment"""
    assignment_id: str
    user_id: str
    role_id: str
    realm_id: Optional[str]
    assigned_by: str
    assigned_at: datetime
    expires_at: Optional[datetime]
    is_active: bool

@dataclass
class SecurityViolation:
    """Represents a security violation event"""
    violation_id: str
    violation_type: str
    severity: str
    user_id: Optional[str]
    source_ip: str
    attempted_action: str
    target_resource: str
    realm_context: str
    detected_at: datetime
    details: Dict[str, Any]

class PromotionManager:
    """Manages knowledge promotion workflows and role-based access control"""
    
    def __init__(self, db_connection):
        self.db = db_connection
        self.logger = logging.getLogger(__name__)
    
    # ===================================================================
    # Knowledge Promotion Methods
    # ===================================================================
    
    def create_promotion_request(self, source_chunk_id: str, target_realm_id: str,
                               promotion_type: PromotionType, justification: str,
                               business_impact: BusinessImpact, requested_by: str,
                               session_id: str) -> str:
        """Create a new promotion request"""
        try:
            cursor = self.db.cursor()
            
            # Call stored procedure to create promotion request
            promotion_id = None
            cursor.callproc('create_promotion_request', [
                source_chunk_id, target_realm_id, promotion_type.value,
                justification, business_impact.value, requested_by, session_id, promotion_id
            ])
            
            # Get the output parameter
            cursor.execute("SELECT @_create_promotion_request_7 AS promotion_id")
            result = cursor.fetchone()
            promotion_id = result[0] if result else None
            
            if not promotion_id:
                raise Exception("Failed to create promotion request")
            
            self.logger.info(f"Created promotion request {promotion_id} for chunk {source_chunk_id}")
            
            # Log audit event
            self._log_audit_event(
                event_type='promotion_requested',
                user_id=requested_by,
                target_type='promotion',
                target_id=promotion_id,
                description=f"Promotion request created for chunk {source_chunk_id} to realm {target_realm_id}",
                session_id=session_id
            )
            
            return promotion_id
            
        except Exception as e:
            self.logger.error(f"Failed to create promotion request: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def approve_promotion_request(self, promotion_id: str, reviewed_by: str,
                                review_notes: str = None) -> bool:
        """Approve a promotion request"""
        try:
            cursor = self.db.cursor()
            
            # Check if user has approval permissions
            if not self._check_promotion_approval_permission(reviewed_by, promotion_id):
                raise PermissionError(f"User {reviewed_by} does not have promotion approval permissions")
            
            # Call stored procedure to approve promotion
            cursor.callproc('approve_promotion_request', [
                promotion_id, reviewed_by, review_notes or ""
            ])
            
            self.logger.info(f"Approved promotion request {promotion_id} by {reviewed_by}")
            
            # Log audit event
            self._log_audit_event(
                event_type='promotion_approved',
                user_id=reviewed_by,
                target_type='promotion',
                target_id=promotion_id,
                description=f"Promotion request approved by {reviewed_by}",
                event_data={'review_notes': review_notes} if review_notes else None
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to approve promotion request {promotion_id}: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def reject_promotion_request(self, promotion_id: str, reviewed_by: str,
                               review_notes: str) -> bool:
        """Reject a promotion request"""
        try:
            cursor = self.db.cursor()
            
            # Check if user has approval permissions
            if not self._check_promotion_approval_permission(reviewed_by, promotion_id):
                raise PermissionError(f"User {reviewed_by} does not have promotion approval permissions")
            
            # Update promotion status
            update_query = """
            UPDATE megamind_promotion_queue 
            SET status = 'rejected', reviewed_by = %s, reviewed_at = NOW(), review_notes = %s
            WHERE promotion_id = %s AND status = 'pending'
            """
            cursor.execute(update_query, (reviewed_by, review_notes, promotion_id))
            
            if cursor.rowcount == 0:
                raise Exception("Promotion request not found or not in pending status")
            
            # Log promotion history
            history_id = f"hist_{uuid.uuid4().hex[:12]}"
            history_query = """
            INSERT INTO megamind_promotion_history 
            (history_id, promotion_id, action_type, action_by, previous_status, new_status, action_reason)
            VALUES (%s, %s, 'rejected', %s, 'pending', 'rejected', %s)
            """
            cursor.execute(history_query, (history_id, promotion_id, reviewed_by, review_notes))
            
            self.db.commit()
            self.logger.info(f"Rejected promotion request {promotion_id} by {reviewed_by}")
            
            # Log audit event
            self._log_audit_event(
                event_type='promotion_rejected',
                user_id=reviewed_by,
                target_type='promotion',
                target_id=promotion_id,
                description=f"Promotion request rejected by {reviewed_by}",
                event_data={'review_notes': review_notes}
            )
            
            return True
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to reject promotion request {promotion_id}: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def get_promotion_requests(self, status: Optional[PromotionStatus] = None,
                             user_id: Optional[str] = None, limit: int = 50) -> List[PromotionRequest]:
        """Get promotion requests with optional filtering"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            query = """
            SELECT pq.*, c.content as original_content
            FROM megamind_promotion_queue pq
            LEFT JOIN megamind_chunks c ON pq.source_chunk_id = c.chunk_id
            WHERE 1=1
            """
            params = []
            
            if status:
                query += " AND pq.status = %s"
                params.append(status.value)
            
            if user_id:
                query += " AND (pq.requested_by = %s OR pq.reviewed_by = %s)"
                params.extend([user_id, user_id])
            
            query += " ORDER BY pq.requested_at DESC LIMIT %s"
            params.append(limit)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            promotions = []
            for row in results:
                promotion = PromotionRequest(
                    promotion_id=row['promotion_id'],
                    source_chunk_id=row['source_chunk_id'],
                    source_realm_id=row['source_realm_id'],
                    target_realm_id=row['target_realm_id'],
                    promotion_type=PromotionType(row['promotion_type']),
                    status=PromotionStatus(row['status']),
                    requested_by=row['requested_by'],
                    requested_at=row['requested_at'],
                    justification=row['justification'],
                    business_impact=BusinessImpact(row['business_impact']),
                    reviewed_by=row.get('reviewed_by'),
                    reviewed_at=row.get('reviewed_at'),
                    review_notes=row.get('review_notes'),
                    target_chunk_id=row.get('target_chunk_id'),
                    original_content=row.get('original_content')
                )
                promotions.append(promotion)
            
            return promotions
            
        except Exception as e:
            self.logger.error(f"Failed to get promotion requests: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_promotion_impact(self, promotion_id: str) -> Optional[PromotionImpact]:
        """Get impact analysis for a promotion request"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            query = """
            SELECT * FROM megamind_promotion_impact 
            WHERE promotion_id = %s
            """
            cursor.execute(query, (promotion_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return PromotionImpact(
                impact_id=row['impact_id'],
                promotion_id=row['promotion_id'],
                affected_chunks_count=row['affected_chunks_count'],
                affected_relationships_count=row['affected_relationships_count'],
                potential_conflicts_count=row['potential_conflicts_count'],
                content_quality_score=float(row['content_quality_score'] or 0),
                relevance_score=float(row['relevance_score'] or 0),
                uniqueness_score=float(row['uniqueness_score'] or 0),
                conflict_analysis=json.loads(row['conflict_analysis']) if row['conflict_analysis'] else None,
                dependency_analysis=json.loads(row['dependency_analysis']) if row['dependency_analysis'] else None,
                usage_impact=json.loads(row['usage_impact']) if row['usage_impact'] else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get promotion impact for {promotion_id}: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Role-Based Access Control Methods
    # ===================================================================
    
    def check_user_permission(self, user_id: str, realm_id: str, 
                            permission: PermissionType) -> bool:
        """Check if user has specific permission in realm"""
        try:
            cursor = self.db.cursor()
            
            # Use the stored function to check permission
            cursor.execute("SELECT check_user_permission(%s, %s, %s) AS has_permission",
                         (user_id, realm_id, permission.value))
            result = cursor.fetchone()
            
            return bool(result[0]) if result else False
            
        except Exception as e:
            self.logger.error(f"Failed to check user permission: {e}")
            return False
        finally:
            if cursor:
                cursor.close()
    
    def assign_user_role(self, user_id: str, role_id: str, realm_id: Optional[str],
                        assigned_by: str, expires_at: Optional[datetime] = None,
                        assignment_reason: str = None) -> str:
        """Assign role to user"""
        try:
            cursor = self.db.cursor()
            
            # Check if assigning user has manage_users permission
            if not self.check_user_permission(assigned_by, realm_id or 'GLOBAL', PermissionType.MANAGE_USERS):
                raise PermissionError(f"User {assigned_by} does not have user management permissions")
            
            assignment_id = f"assign_{uuid.uuid4().hex[:12]}"
            
            query = """
            INSERT INTO megamind_user_role_assignments 
            (assignment_id, user_id, role_id, realm_id, assigned_by, expires_at, assignment_reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(query, (assignment_id, user_id, role_id, realm_id, 
                                 assigned_by, expires_at, assignment_reason))
            self.db.commit()
            
            self.logger.info(f"Assigned role {role_id} to user {user_id} in realm {realm_id}")
            
            # Log audit event
            self._log_audit_event(
                event_type='role_assigned',
                user_id=assigned_by,
                target_type='user',
                target_id=user_id,
                target_realm_id=realm_id,
                description=f"Role {role_id} assigned to user {user_id}",
                event_data={'role_id': role_id, 'realm_id': realm_id}
            )
            
            return assignment_id
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to assign role: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def revoke_user_role(self, assignment_id: str, revoked_by: str, 
                        revocation_reason: str = None) -> bool:
        """Revoke user role assignment"""
        try:
            cursor = self.db.cursor()
            
            # Get assignment details for permission check
            query = "SELECT user_id, realm_id FROM megamind_user_role_assignments WHERE assignment_id = %s"
            cursor.execute(query, (assignment_id,))
            result = cursor.fetchone()
            
            if not result:
                raise Exception("Role assignment not found")
            
            user_id, realm_id = result
            
            # Check if revoking user has manage_users permission
            if not self.check_user_permission(revoked_by, realm_id or 'GLOBAL', PermissionType.MANAGE_USERS):
                raise PermissionError(f"User {revoked_by} does not have user management permissions")
            
            # Deactivate assignment
            update_query = """
            UPDATE megamind_user_role_assignments 
            SET is_active = FALSE 
            WHERE assignment_id = %s
            """
            cursor.execute(update_query, (assignment_id,))
            
            if cursor.rowcount == 0:
                raise Exception("Role assignment not found or already revoked")
            
            self.db.commit()
            self.logger.info(f"Revoked role assignment {assignment_id} by {revoked_by}")
            
            # Log audit event
            self._log_audit_event(
                event_type='role_removed',
                user_id=revoked_by,
                target_type='user',
                target_id=user_id,
                target_realm_id=realm_id,
                description=f"Role assignment {assignment_id} revoked",
                event_data={'assignment_id': assignment_id, 'reason': revocation_reason}
            )
            
            return True
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to revoke role assignment: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def get_user_roles(self, user_id: str, realm_id: Optional[str] = None) -> List[UserRole]:
        """Get user's role assignments"""
        try:
            cursor = self.db.cursor(dictionary=True)
            
            query = """
            SELECT * FROM megamind_user_role_assignments 
            WHERE user_id = %s AND is_active = TRUE 
            AND (expires_at IS NULL OR expires_at > NOW())
            """
            params = [user_id]
            
            if realm_id:
                query += " AND (realm_id = %s OR realm_id IS NULL)"
                params.append(realm_id)
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            roles = []
            for row in results:
                role = UserRole(
                    assignment_id=row['assignment_id'],
                    user_id=row['user_id'],
                    role_id=row['role_id'],
                    realm_id=row['realm_id'],
                    assigned_by=row['assigned_by'],
                    assigned_at=row['assigned_at'],
                    expires_at=row['expires_at'],
                    is_active=row['is_active']
                )
                roles.append(role)
            
            return roles
            
        except Exception as e:
            self.logger.error(f"Failed to get user roles: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    # ===================================================================
    # Security and Audit Methods
    # ===================================================================
    
    def log_security_violation(self, violation_type: str, severity: str,
                             user_id: Optional[str], source_ip: str,
                             attempted_action: str, target_resource: str = None,
                             realm_context: str = None) -> str:
        """Log a security violation"""
        try:
            cursor = self.db.cursor()
            
            violation_id = f"violation_{uuid.uuid4().hex[:12]}"
            
            query = """
            INSERT INTO megamind_security_violations 
            (violation_id, violation_type, severity, user_id, source_ip, 
             attempted_action, target_resource, realm_context, response_action)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'logged')
            """
            
            cursor.execute(query, (violation_id, violation_type, severity, user_id,
                                 source_ip, attempted_action, target_resource, realm_context))
            self.db.commit()
            
            self.logger.warning(f"Security violation logged: {violation_type} by {user_id or 'unknown'} from {source_ip}")
            
            return violation_id
            
        except Exception as e:
            self.db.rollback()
            self.logger.error(f"Failed to log security violation: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def _log_audit_event(self, event_type: str, user_id: str, target_type: str,
                        target_id: str, description: str, target_realm_id: str = None,
                        event_data: Dict = None, security_level: str = 'low',
                        session_id: str = None):
        """Log an audit event"""
        try:
            cursor = self.db.cursor()
            
            audit_id = f"audit_{uuid.uuid4().hex[:12]}"
            
            query = """
            INSERT INTO megamind_audit_log 
            (audit_id, event_type, event_category, user_id, target_type, target_id,
             target_realm_id, event_description, event_data, security_level, session_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            # Determine event category based on event type
            if 'promotion' in event_type:
                category = 'promotion'
            elif 'role' in event_type:
                category = 'security'
            elif 'realm' in event_type:
                category = 'administration'
            else:
                category = 'data'
            
            cursor.execute(query, (
                audit_id, event_type, category, user_id, target_type, target_id,
                target_realm_id, description, json.dumps(event_data) if event_data else None,
                security_level, session_id
            ))
            
            self.db.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            # Don't raise exception for audit logging failures
    
    def _check_promotion_approval_permission(self, user_id: str, promotion_id: str) -> bool:
        """Check if user can approve specific promotion request"""
        try:
            cursor = self.db.cursor()
            
            # Get target realm for the promotion
            cursor.execute("SELECT target_realm_id FROM megamind_promotion_queue WHERE promotion_id = %s", 
                         (promotion_id,))
            result = cursor.fetchone()
            
            if not result:
                return False
            
            target_realm_id = result[0]
            
            # Check if user has approval permission in target realm
            return self.check_user_permission(user_id, target_realm_id, PermissionType.APPROVE_PROMOTIONS)
            
        except Exception as e:
            self.logger.error(f"Failed to check promotion approval permission: {e}")
            return False
        finally:
            if cursor:
                cursor.close()