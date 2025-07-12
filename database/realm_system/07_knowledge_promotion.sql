-- ===================================================================
-- MegaMind Context Database: Knowledge Promotion System (Phase 3)
-- Advanced features for promoting knowledge between realms
-- ===================================================================

-- Knowledge Promotion Queue for approval workflows
CREATE TABLE megamind_promotion_queue (
    promotion_id VARCHAR(50) PRIMARY KEY,
    source_chunk_id VARCHAR(50) NOT NULL,
    source_realm_id VARCHAR(50) NOT NULL,
    target_realm_id VARCHAR(50) NOT NULL,
    promotion_type ENUM('copy', 'move', 'reference') NOT NULL DEFAULT 'copy',
    
    -- Approval workflow
    status ENUM('pending', 'approved', 'rejected', 'processing', 'completed') NOT NULL DEFAULT 'pending',
    requested_by VARCHAR(100) NOT NULL,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_by VARCHAR(100) DEFAULT NULL,
    reviewed_at TIMESTAMP DEFAULT NULL,
    completed_at TIMESTAMP DEFAULT NULL,
    
    -- Justification and context
    justification TEXT NOT NULL,
    business_impact ENUM('low', 'medium', 'high', 'critical') NOT NULL DEFAULT 'medium',
    review_notes TEXT DEFAULT NULL,
    
    -- Promotion metadata
    original_content TEXT NOT NULL, -- Snapshot of content at promotion time
    target_chunk_id VARCHAR(50) DEFAULT NULL, -- Set after successful promotion
    promotion_session_id VARCHAR(50) NOT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (source_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (source_realm_id) REFERENCES megamind_realms(realm_id),
    FOREIGN KEY (target_realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Indexes for workflow management
    INDEX idx_promotion_status (status, requested_at DESC),
    INDEX idx_promotion_reviewer (reviewed_by, reviewed_at DESC),
    INDEX idx_promotion_source (source_realm_id, source_chunk_id),
    INDEX idx_promotion_target (target_realm_id, status),
    INDEX idx_promotion_session (promotion_session_id, status)
) ENGINE=InnoDB;

-- Knowledge Promotion History for audit trail
CREATE TABLE megamind_promotion_history (
    history_id VARCHAR(50) PRIMARY KEY,
    promotion_id VARCHAR(50) NOT NULL,
    
    -- Action tracking
    action_type ENUM('created', 'approved', 'rejected', 'completed', 'failed', 'cancelled') NOT NULL,
    action_by VARCHAR(100) NOT NULL,
    action_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    action_reason TEXT DEFAULT NULL,
    
    -- State snapshot
    previous_status ENUM('pending', 'approved', 'rejected', 'processing', 'completed') DEFAULT NULL,
    new_status ENUM('pending', 'approved', 'rejected', 'processing', 'completed') NOT NULL,
    
    -- Additional context
    system_metadata JSON DEFAULT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (promotion_id) REFERENCES megamind_promotion_queue(promotion_id) ON DELETE CASCADE,
    
    -- Indexes for audit queries
    INDEX idx_history_promotion (promotion_id, action_at DESC),
    INDEX idx_history_user (action_by, action_at DESC),
    INDEX idx_history_action (action_type, action_at DESC)
) ENGINE=InnoDB;

-- Promotion Impact Analysis for review assistance
CREATE TABLE megamind_promotion_impact (
    impact_id VARCHAR(50) PRIMARY KEY,
    promotion_id VARCHAR(50) NOT NULL,
    
    -- Impact metrics
    affected_chunks_count INT DEFAULT 0,
    affected_relationships_count INT DEFAULT 0,
    potential_conflicts_count INT DEFAULT 0,
    
    -- Analysis results
    conflict_analysis JSON DEFAULT NULL, -- Details of potential conflicts
    dependency_analysis JSON DEFAULT NULL, -- Dependency chains affected
    usage_impact JSON DEFAULT NULL, -- Usage pattern analysis
    
    -- Quality assessment
    content_quality_score DECIMAL(3,2) DEFAULT NULL, -- 0.00 to 1.00
    relevance_score DECIMAL(3,2) DEFAULT NULL, -- 0.00 to 1.00
    uniqueness_score DECIMAL(3,2) DEFAULT NULL, -- 0.00 to 1.00
    
    -- Analysis metadata
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    analysis_version VARCHAR(20) DEFAULT '1.0',
    
    -- Foreign key constraints
    FOREIGN KEY (promotion_id) REFERENCES megamind_promotion_queue(promotion_id) ON DELETE CASCADE,
    
    -- Indexes for analysis queries
    INDEX idx_impact_promotion (promotion_id),
    INDEX idx_impact_quality (content_quality_score DESC, relevance_score DESC),
    INDEX idx_impact_conflicts (potential_conflicts_count DESC, analyzed_at DESC)
) ENGINE=InnoDB;

-- Role-Based Access Control Tables

-- Realm Roles Definition
CREATE TABLE megamind_realm_roles (
    role_id VARCHAR(50) PRIMARY KEY,
    role_name VARCHAR(100) NOT NULL,
    role_type ENUM('global', 'realm_specific', 'project_specific') NOT NULL,
    description TEXT,
    
    -- Permissions
    can_read BOOLEAN DEFAULT TRUE,
    can_write BOOLEAN DEFAULT FALSE,
    can_delete BOOLEAN DEFAULT FALSE,
    can_promote BOOLEAN DEFAULT FALSE,
    can_approve_promotions BOOLEAN DEFAULT FALSE,
    can_manage_users BOOLEAN DEFAULT FALSE,
    can_manage_realm BOOLEAN DEFAULT FALSE,
    
    -- Role hierarchy
    parent_role_id VARCHAR(50) DEFAULT NULL,
    is_system_role BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Foreign key constraints
    FOREIGN KEY (parent_role_id) REFERENCES megamind_realm_roles(role_id),
    
    -- Unique role names
    UNIQUE KEY unique_role_name (role_name),
    
    -- Indexes
    INDEX idx_roles_type (role_type, is_active),
    INDEX idx_roles_permissions (can_promote, can_approve_promotions),
    INDEX idx_roles_hierarchy (parent_role_id, is_active)
) ENGINE=InnoDB;

-- User Role Assignments
CREATE TABLE megamind_user_role_assignments (
    assignment_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    role_id VARCHAR(50) NOT NULL,
    realm_id VARCHAR(50), -- NULL for global roles
    
    -- Assignment metadata
    assigned_by VARCHAR(100) NOT NULL,
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Assignment context
    assignment_reason TEXT DEFAULT NULL,
    assignment_session_id VARCHAR(50) DEFAULT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (role_id) REFERENCES megamind_realm_roles(role_id),
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Prevent duplicate active assignments
    UNIQUE KEY unique_user_role_realm (user_id, role_id, realm_id, is_active),
    
    -- Indexes
    INDEX idx_assignments_user (user_id, is_active, expires_at),
    INDEX idx_assignments_role (role_id, realm_id, is_active),
    INDEX idx_assignments_realm (realm_id, is_active),
    INDEX idx_assignments_expiry (expires_at, is_active)
) ENGINE=InnoDB;

-- Audit Logging System

-- Comprehensive Audit Log
CREATE TABLE megamind_audit_log (
    audit_id VARCHAR(50) PRIMARY KEY,
    
    -- Event identification
    event_type ENUM('chunk_created', 'chunk_updated', 'chunk_deleted', 'chunk_accessed', 
                    'relationship_created', 'relationship_deleted', 'promotion_requested', 
                    'promotion_approved', 'promotion_rejected', 'role_assigned', 'role_removed',
                    'realm_created', 'realm_updated', 'user_login', 'user_logout', 'permission_denied') NOT NULL,
    event_category ENUM('data', 'security', 'administration', 'promotion', 'access') NOT NULL,
    
    -- Actor information
    user_id VARCHAR(100) NOT NULL,
    user_role VARCHAR(100) DEFAULT NULL,
    source_ip VARCHAR(45) DEFAULT NULL, -- IPv6 compatible
    user_agent TEXT DEFAULT NULL,
    
    -- Target information
    target_type ENUM('chunk', 'relationship', 'realm', 'user', 'role', 'promotion') NOT NULL,
    target_id VARCHAR(50) NOT NULL,
    target_realm_id VARCHAR(50) DEFAULT NULL,
    
    -- Event details
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_description TEXT NOT NULL,
    event_data JSON DEFAULT NULL, -- Detailed event information
    
    -- Security context
    session_id VARCHAR(50) DEFAULT NULL,
    request_id VARCHAR(50) DEFAULT NULL,
    
    -- Impact assessment
    security_level ENUM('low', 'medium', 'high', 'critical') NOT NULL DEFAULT 'low',
    data_sensitivity ENUM('public', 'internal', 'confidential', 'restricted') NOT NULL DEFAULT 'internal',
    
    -- Foreign key constraints
    FOREIGN KEY (target_realm_id) REFERENCES megamind_realms(realm_id),
    
    -- Indexes for audit queries
    INDEX idx_audit_user (user_id, event_timestamp DESC),
    INDEX idx_audit_event (event_type, event_category, event_timestamp DESC),
    INDEX idx_audit_target (target_type, target_id, event_timestamp DESC),
    INDEX idx_audit_realm (target_realm_id, event_timestamp DESC),
    INDEX idx_audit_security (security_level, event_timestamp DESC),
    INDEX idx_audit_session (session_id, event_timestamp DESC)
) ENGINE=InnoDB;

-- Security Violation Tracking
CREATE TABLE megamind_security_violations (
    violation_id VARCHAR(50) PRIMARY KEY,
    
    -- Violation details
    violation_type ENUM('unauthorized_access', 'permission_escalation', 'data_breach_attempt', 
                        'injection_attempt', 'brute_force', 'suspicious_pattern') NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') NOT NULL DEFAULT 'medium',
    
    -- Actor information
    user_id VARCHAR(100) DEFAULT NULL,
    source_ip VARCHAR(45) NOT NULL,
    user_agent TEXT DEFAULT NULL,
    
    -- Context
    attempted_action TEXT NOT NULL,
    target_resource VARCHAR(255) DEFAULT NULL,
    realm_context VARCHAR(50) DEFAULT NULL,
    
    -- Detection
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_method ENUM('automatic', 'manual', 'system_alert') NOT NULL DEFAULT 'automatic',
    
    -- Response
    response_action ENUM('logged', 'blocked', 'rate_limited', 'account_suspended') NOT NULL DEFAULT 'logged',
    response_notes TEXT DEFAULT NULL,
    
    -- Investigation
    investigated BOOLEAN DEFAULT FALSE,
    investigation_notes TEXT DEFAULT NULL,
    false_positive BOOLEAN DEFAULT FALSE,
    
    -- Foreign key constraints
    FOREIGN KEY (realm_context) REFERENCES megamind_realms(realm_id),
    
    -- Indexes for security analysis
    INDEX idx_violations_severity (severity, detected_at DESC),
    INDEX idx_violations_user (user_id, detected_at DESC),
    INDEX idx_violations_ip (source_ip, detected_at DESC),
    INDEX idx_violations_type (violation_type, severity, detected_at DESC),
    INDEX idx_violations_investigation (investigated, false_positive, detected_at DESC)
) ENGINE=InnoDB;

-- ===================================================================
-- Stored Procedures for Knowledge Promotion
-- ===================================================================

DELIMITER //

-- Create Promotion Request
CREATE PROCEDURE create_promotion_request(
    IN p_source_chunk_id VARCHAR(50),
    IN p_target_realm_id VARCHAR(50),
    IN p_promotion_type ENUM('copy', 'move', 'reference'),
    IN p_justification TEXT,
    IN p_business_impact ENUM('low', 'medium', 'high', 'critical'),
    IN p_requested_by VARCHAR(100),
    IN p_session_id VARCHAR(50),
    OUT p_promotion_id VARCHAR(50)
)
BEGIN
    DECLARE v_source_realm_id VARCHAR(50);
    DECLARE v_content TEXT;
    DECLARE v_promotion_id VARCHAR(50);
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;
    
    START TRANSACTION;
    
    -- Get source chunk information
    SELECT realm_id, content INTO v_source_realm_id, v_content
    FROM megamind_chunks 
    WHERE chunk_id = p_source_chunk_id;
    
    IF v_source_realm_id IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Source chunk not found';
    END IF;
    
    -- Validate promotion is not from/to same realm
    IF v_source_realm_id = p_target_realm_id THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Cannot promote chunk within same realm';
    END IF;
    
    -- Generate promotion ID
    SET v_promotion_id = CONCAT('promo_', SUBSTRING(MD5(CONCAT(p_source_chunk_id, p_target_realm_id, NOW())), 1, 12));
    
    -- Create promotion request
    INSERT INTO megamind_promotion_queue (
        promotion_id, source_chunk_id, source_realm_id, target_realm_id,
        promotion_type, justification, business_impact, requested_by,
        promotion_session_id, original_content
    ) VALUES (
        v_promotion_id, p_source_chunk_id, v_source_realm_id, p_target_realm_id,
        p_promotion_type, p_justification, p_business_impact, p_requested_by,
        p_session_id, v_content
    );
    
    -- Log promotion history
    INSERT INTO megamind_promotion_history (
        history_id, promotion_id, action_type, action_by, new_status, action_reason
    ) VALUES (
        CONCAT('hist_', SUBSTRING(MD5(CONCAT(v_promotion_id, 'created', NOW())), 1, 12)),
        v_promotion_id, 'created', p_requested_by, 'pending', 'Promotion request created'
    );
    
    -- Run impact analysis
    CALL analyze_promotion_impact(v_promotion_id);
    
    SET p_promotion_id = v_promotion_id;
    
    COMMIT;
END //

-- Approve Promotion Request
CREATE PROCEDURE approve_promotion_request(
    IN p_promotion_id VARCHAR(50),
    IN p_reviewed_by VARCHAR(100),
    IN p_review_notes TEXT
)
BEGIN
    DECLARE v_source_chunk_id VARCHAR(50);
    DECLARE v_target_realm_id VARCHAR(50);
    DECLARE v_promotion_type ENUM('copy', 'move', 'reference');
    DECLARE v_content TEXT;
    DECLARE v_source_document VARCHAR(255);
    DECLARE v_section_path VARCHAR(500);
    DECLARE v_chunk_type ENUM('rule', 'function', 'section', 'example');
    DECLARE v_new_chunk_id VARCHAR(50);
    DECLARE EXIT HANDLER FOR SQLEXCEPTION
    BEGIN
        ROLLBACK;
        RESIGNAL;
    END;
    
    START TRANSACTION;
    
    -- Get promotion details
    SELECT pq.source_chunk_id, pq.target_realm_id, pq.promotion_type, pq.original_content,
           c.source_document, c.section_path, c.chunk_type
    INTO v_source_chunk_id, v_target_realm_id, v_promotion_type, v_content,
         v_source_document, v_section_path, v_chunk_type
    FROM megamind_promotion_queue pq
    JOIN megamind_chunks c ON pq.source_chunk_id = c.chunk_id
    WHERE pq.promotion_id = p_promotion_id AND pq.status = 'pending';
    
    IF v_source_chunk_id IS NULL THEN
        SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Promotion request not found or not in pending status';
    END IF;
    
    -- Update promotion status
    UPDATE megamind_promotion_queue 
    SET status = 'approved', reviewed_by = p_reviewed_by, reviewed_at = NOW(), review_notes = p_review_notes
    WHERE promotion_id = p_promotion_id;
    
    -- Log approval
    INSERT INTO megamind_promotion_history (
        history_id, promotion_id, action_type, action_by, previous_status, new_status, action_reason
    ) VALUES (
        CONCAT('hist_', SUBSTRING(MD5(CONCAT(p_promotion_id, 'approved', NOW())), 1, 12)),
        p_promotion_id, 'approved', p_reviewed_by, 'pending', 'approved', p_review_notes
    );
    
    -- Execute promotion based on type
    IF v_promotion_type = 'copy' THEN
        -- Create new chunk in target realm
        SET v_new_chunk_id = CONCAT('chunk_', SUBSTRING(MD5(CONCAT(v_content, v_target_realm_id, NOW())), 1, 12));
        
        INSERT INTO megamind_chunks (
            chunk_id, content, source_document, section_path, chunk_type,
            line_count, realm_id, token_count, access_count
        ) 
        SELECT v_new_chunk_id, content, source_document, section_path, chunk_type,
               line_count, v_target_realm_id, token_count, 1
        FROM megamind_chunks 
        WHERE chunk_id = v_source_chunk_id;
        
        -- Update promotion with target chunk ID
        UPDATE megamind_promotion_queue 
        SET target_chunk_id = v_new_chunk_id, status = 'completed', completed_at = NOW()
        WHERE promotion_id = p_promotion_id;
        
    ELSEIF v_promotion_type = 'move' THEN
        -- Move chunk to target realm
        UPDATE megamind_chunks 
        SET realm_id = v_target_realm_id 
        WHERE chunk_id = v_source_chunk_id;
        
        -- Update promotion
        UPDATE megamind_promotion_queue 
        SET target_chunk_id = v_source_chunk_id, status = 'completed', completed_at = NOW()
        WHERE promotion_id = p_promotion_id;
        
    ELSEIF v_promotion_type = 'reference' THEN
        -- Create reference relationship (implementation depends on relationship system)
        -- For now, just mark as completed
        UPDATE megamind_promotion_queue 
        SET status = 'completed', completed_at = NOW()
        WHERE promotion_id = p_promotion_id;
    END IF;
    
    -- Log completion
    INSERT INTO megamind_promotion_history (
        history_id, promotion_id, action_type, action_by, previous_status, new_status, action_reason
    ) VALUES (
        CONCAT('hist_', SUBSTRING(MD5(CONCAT(p_promotion_id, 'completed', NOW())), 1, 12)),
        p_promotion_id, 'completed', p_reviewed_by, 'approved', 'completed', 'Promotion executed successfully'
    );
    
    COMMIT;
END //

-- Analyze Promotion Impact
CREATE PROCEDURE analyze_promotion_impact(
    IN p_promotion_id VARCHAR(50)
)
BEGIN
    DECLARE v_source_chunk_id VARCHAR(50);
    DECLARE v_target_realm_id VARCHAR(50);
    DECLARE v_affected_chunks INT DEFAULT 0;
    DECLARE v_affected_relationships INT DEFAULT 0;
    DECLARE v_potential_conflicts INT DEFAULT 0;
    DECLARE v_impact_id VARCHAR(50);
    
    -- Get promotion details
    SELECT source_chunk_id, target_realm_id 
    INTO v_source_chunk_id, v_target_realm_id
    FROM megamind_promotion_queue 
    WHERE promotion_id = p_promotion_id;
    
    -- Count affected relationships
    SELECT COUNT(*) INTO v_affected_relationships
    FROM megamind_chunk_relationships 
    WHERE chunk_id = v_source_chunk_id OR related_chunk_id = v_source_chunk_id;
    
    -- Check for potential conflicts (chunks with similar content in target realm)
    SELECT COUNT(*) INTO v_potential_conflicts
    FROM megamind_chunks c1
    JOIN megamind_chunks c2 ON c1.source_document = c2.source_document AND c1.section_path = c2.section_path
    WHERE c1.chunk_id = v_source_chunk_id AND c2.realm_id = v_target_realm_id;
    
    -- Generate impact ID
    SET v_impact_id = CONCAT('impact_', SUBSTRING(MD5(CONCAT(p_promotion_id, NOW())), 1, 12));
    
    -- Store impact analysis
    INSERT INTO megamind_promotion_impact (
        impact_id, promotion_id, affected_chunks_count, affected_relationships_count,
        potential_conflicts_count, content_quality_score, relevance_score, uniqueness_score
    ) VALUES (
        v_impact_id, p_promotion_id, v_affected_chunks, v_affected_relationships,
        v_potential_conflicts, 0.80, 0.75, 0.85  -- Default scores, would be calculated by AI in real implementation
    );
    
END //

-- Check User Permissions
CREATE FUNCTION check_user_permission(
    p_user_id VARCHAR(100),
    p_realm_id VARCHAR(50),
    p_permission_type ENUM('read', 'write', 'delete', 'promote', 'approve_promotions', 'manage_users', 'manage_realm')
) RETURNS BOOLEAN
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE v_has_permission BOOLEAN DEFAULT FALSE;
    
    -- Check if user has required permission through any active role assignment
    SELECT COUNT(*) > 0 INTO v_has_permission
    FROM megamind_user_role_assignments ura
    JOIN megamind_realm_roles rr ON ura.role_id = rr.role_id
    WHERE ura.user_id = p_user_id 
      AND (ura.realm_id = p_realm_id OR ura.realm_id IS NULL) -- NULL means global role
      AND ura.is_active = TRUE
      AND (ura.expires_at IS NULL OR ura.expires_at > NOW())
      AND rr.is_active = TRUE
      AND (
          (p_permission_type = 'read' AND rr.can_read = TRUE) OR
          (p_permission_type = 'write' AND rr.can_write = TRUE) OR
          (p_permission_type = 'delete' AND rr.can_delete = TRUE) OR
          (p_permission_type = 'promote' AND rr.can_promote = TRUE) OR
          (p_permission_type = 'approve_promotions' AND rr.can_approve_promotions = TRUE) OR
          (p_permission_type = 'manage_users' AND rr.can_manage_users = TRUE) OR
          (p_permission_type = 'manage_realm' AND rr.can_manage_realm = TRUE)
      );
    
    RETURN v_has_permission;
END //

DELIMITER ;

-- ===================================================================
-- Views for Knowledge Promotion Management
-- ===================================================================

-- Promotion Dashboard View
CREATE VIEW megamind_promotion_dashboard AS
SELECT 
    pq.promotion_id,
    pq.source_chunk_id,
    c.content AS source_content,
    c.source_document,
    sr.realm_name AS source_realm_name,
    tr.realm_name AS target_realm_name,
    pq.promotion_type,
    pq.status,
    pq.business_impact,
    pq.requested_by,
    pq.requested_at,
    pq.reviewed_by,
    pq.reviewed_at,
    pq.justification,
    pi.content_quality_score,
    pi.potential_conflicts_count,
    DATEDIFF(NOW(), pq.requested_at) AS days_pending
FROM megamind_promotion_queue pq
JOIN megamind_chunks c ON pq.source_chunk_id = c.chunk_id
JOIN megamind_realms sr ON pq.source_realm_id = sr.realm_id
JOIN megamind_realms tr ON pq.target_realm_id = tr.realm_id
LEFT JOIN megamind_promotion_impact pi ON pq.promotion_id = pi.promotion_id
ORDER BY 
    CASE pq.business_impact 
        WHEN 'critical' THEN 1 
        WHEN 'high' THEN 2 
        WHEN 'medium' THEN 3 
        WHEN 'low' THEN 4 
    END,
    pq.requested_at DESC;

-- Security Overview View
CREATE VIEW megamind_security_overview AS
SELECT 
    DATE(detected_at) AS violation_date,
    violation_type,
    severity,
    COUNT(*) AS incident_count,
    COUNT(DISTINCT user_id) AS unique_users_affected,
    COUNT(DISTINCT source_ip) AS unique_ips,
    SUM(CASE WHEN false_positive = TRUE THEN 1 ELSE 0 END) AS false_positives
FROM megamind_security_violations 
WHERE detected_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY DATE(detected_at), violation_type, severity
ORDER BY violation_date DESC, 
         CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 WHEN 'low' THEN 4 END;

-- ===================================================================
-- Initial System Roles and Permissions
-- ===================================================================

-- Insert default system roles
INSERT INTO megamind_realm_roles (role_id, role_name, role_type, description, 
                                  can_read, can_write, can_delete, can_promote, can_approve_promotions, 
                                  can_manage_users, can_manage_realm, is_system_role, created_by) VALUES
('role_global_admin', 'Global Administrator', 'global', 'Full system administration privileges', 
 TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, 'system'),
('role_realm_admin', 'Realm Administrator', 'realm_specific', 'Full administration within assigned realm', 
 TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, 'system'),
('role_contributor', 'Contributor', 'realm_specific', 'Create and modify content within assigned realm', 
 TRUE, TRUE, FALSE, TRUE, FALSE, FALSE, FALSE, TRUE, 'system'),
('role_reviewer', 'Reviewer', 'realm_specific', 'Review and approve promotions', 
 TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, 'system'),
('role_viewer', 'Viewer', 'realm_specific', 'Read-only access to realm content', 
 TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, 'system');