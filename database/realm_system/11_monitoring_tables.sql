-- =====================================================================
-- MegaMind Context Database - Monitoring and Dashboard Tables
-- =====================================================================
-- Additional tables for monitoring, alerting, and dashboard functionality
-- Created: 2025-07-12
-- Purpose: Support Phase 4 monitoring and dashboard features

-- =====================================================================
-- Monitoring Rules Configuration
-- =====================================================================

CREATE TABLE IF NOT EXISTS megamind_monitoring_rules (
    rule_id VARCHAR(50) PRIMARY KEY,
    component_type ENUM('database', 'inheritance', 'search', 'promotion', 'security', 'realm') NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    operator ENUM('gt', 'lt', 'eq', 'ne') NOT NULL,
    threshold_value DECIMAL(10,2) NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    check_interval_minutes INT NOT NULL DEFAULT 5,
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_monitoring_rules_component (component_type, enabled),
    INDEX idx_monitoring_rules_severity (severity, enabled)
) ENGINE=InnoDB;

-- =====================================================================
-- Monitoring Alerts
-- =====================================================================

CREATE TABLE IF NOT EXISTS megamind_monitoring_alerts (
    alert_id VARCHAR(50) PRIMARY KEY,
    realm_id VARCHAR(50) DEFAULT NULL,
    component_type ENUM('database', 'inheritance', 'search', 'promotion', 'security', 'realm') NOT NULL,
    component_name VARCHAR(100) NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    triggered_at TIMESTAMP NOT NULL,
    resolved_at TIMESTAMP DEFAULT NULL,
    acknowledged_at TIMESTAMP DEFAULT NULL,
    acknowledged_by VARCHAR(100) DEFAULT NULL,
    resolution_notes TEXT DEFAULT NULL,
    
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE SET NULL,
    
    INDEX idx_alerts_status (resolved_at, severity, triggered_at DESC),
    INDEX idx_alerts_component (component_type, component_name, triggered_at DESC),
    INDEX idx_alerts_realm (realm_id, severity, triggered_at DESC),
    INDEX idx_alerts_severity (severity, triggered_at DESC)
) ENGINE=InnoDB;

-- =====================================================================
-- Dashboard Configurations
-- =====================================================================

CREATE TABLE IF NOT EXISTS megamind_dashboards (
    dashboard_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    realm_id VARCHAR(50) DEFAULT NULL,
    layout_config JSON NOT NULL,
    created_by VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_public BOOLEAN DEFAULT FALSE,
    
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE SET NULL,
    
    INDEX idx_dashboards_realm (realm_id, is_public),
    INDEX idx_dashboards_created (created_by, created_at DESC),
    INDEX idx_dashboards_public (is_public, created_at DESC)
) ENGINE=InnoDB;

-- =====================================================================
-- Performance Insights Storage
-- =====================================================================

CREATE TABLE IF NOT EXISTS megamind_performance_insights (
    insight_id VARCHAR(50) PRIMARY KEY,
    realm_id VARCHAR(50) NOT NULL,
    insight_type ENUM('engagement', 'relationships', 'distribution', 'content_health', 'performance') NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    recommendation TEXT NOT NULL,
    impact_score DECIMAL(3,2) NOT NULL CHECK (impact_score >= 0.0 AND impact_score <= 1.0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT NULL,
    addressed_at TIMESTAMP DEFAULT NULL,
    addressed_by VARCHAR(100) DEFAULT NULL,
    
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    
    INDEX idx_insights_realm (realm_id, severity, created_at DESC),
    INDEX idx_insights_type (insight_type, severity),
    INDEX idx_insights_active (expires_at, addressed_at, created_at DESC)
) ENGINE=InnoDB;

-- =====================================================================
-- Insert Default Monitoring Rules
-- =====================================================================

INSERT INTO megamind_monitoring_rules (rule_id, component_type, metric_name, operator, threshold_value, severity, check_interval_minutes) VALUES
-- Database monitoring rules
('rule_db_health', 'database', 'health_score', 'lt', 80.0, 'medium', 5),
('rule_db_critical', 'database', 'health_score', 'lt', 50.0, 'critical', 1),
('rule_db_response', 'database', 'response_time_ms', 'gt', 2000.0, 'medium', 5),
('rule_db_connections', 'database', 'active_connections', 'gt', 80.0, 'high', 5),

-- Realm monitoring rules
('rule_realm_health', 'realm', 'health_score', 'lt', 70.0, 'medium', 15),
('rule_realm_critical', 'realm', 'health_score', 'lt', 40.0, 'critical', 10),
('rule_realm_empty', 'realm', 'chunk_count', 'eq', 0.0, 'medium', 60),
('rule_realm_stagnant', 'realm', 'recent_access_count', 'eq', 0.0, 'low', 120),

-- Search monitoring rules
('rule_search_slow', 'search', 'search_time_ms', 'gt', 3000.0, 'medium', 10),
('rule_search_critical', 'search', 'search_time_ms', 'gt', 5000.0, 'high', 5),
('rule_search_no_results', 'search', 'test_results_found', 'eq', 0.0, 'medium', 30),

-- Inheritance monitoring rules
('rule_inherit_health', 'inheritance', 'health_score', 'lt', 80.0, 'medium', 30),
('rule_inherit_no_setup', 'inheritance', 'total_inheritance_relationships', 'eq', 0.0, 'medium', 60),

-- Promotion monitoring rules
('rule_promotion_backlog', 'promotion', 'pending_requests', 'gt', 10.0, 'medium', 20),
('rule_promotion_critical', 'promotion', 'pending_requests', 'gt', 25.0, 'high', 10),
('rule_promotion_slow', 'promotion', 'avg_processing_hours', 'gt', 72.0, 'medium', 60),

-- Security monitoring rules
('rule_security_violations', 'security', 'critical_violations_24h', 'gt', 0.0, 'critical', 5),
('rule_security_high_violations', 'security', 'high_violations_24h', 'gt', 3.0, 'high', 10),
('rule_security_failed_attempts', 'security', 'failed_attempts_1h', 'gt', 20.0, 'medium', 15);

-- =====================================================================
-- Monitoring Views for Dashboard Data
-- =====================================================================

-- Real-time alert summary view
CREATE OR REPLACE VIEW megamind_alert_summary AS
SELECT 
    component_type,
    severity,
    COUNT(*) as alert_count,
    MAX(triggered_at) as latest_alert,
    MIN(triggered_at) as earliest_alert
FROM megamind_monitoring_alerts
WHERE resolved_at IS NULL
GROUP BY component_type, severity
ORDER BY 
    CASE severity 
        WHEN 'critical' THEN 1
        WHEN 'high' THEN 2
        WHEN 'medium' THEN 3
        WHEN 'low' THEN 4
    END,
    alert_count DESC;

-- System health overview
CREATE OR REPLACE VIEW megamind_system_health_overview AS
SELECT 
    component_type,
    COUNT(*) as total_components,
    COUNT(CASE WHEN status = 'healthy' THEN 1 END) as healthy_count,
    COUNT(CASE WHEN status = 'warning' THEN 1 END) as warning_count,
    COUNT(CASE WHEN status = 'critical' THEN 1 END) as critical_count,
    COUNT(CASE WHEN status = 'down' THEN 1 END) as down_count,
    AVG(health_score) as avg_health_score,
    MIN(health_score) as min_health_score,
    MAX(last_check) as last_check
FROM megamind_system_health
GROUP BY component_type
ORDER BY avg_health_score DESC;

-- Realm performance summary
CREATE OR REPLACE VIEW megamind_realm_performance_summary AS
SELECT 
    r.realm_id,
    r.realm_name,
    r.realm_type,
    COUNT(c.chunk_id) as chunk_count,
    AVG(c.access_count) as avg_access_count,
    COUNT(CASE WHEN c.last_accessed >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as recent_activity,
    COUNT(cr.relationship_id) as relationship_count,
    COUNT(DISTINCT ct.tag_value) as subsystem_count,
    sh.health_score,
    sh.status as health_status,
    COUNT(ma.alert_id) as active_alerts
FROM megamind_realms r
LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id
LEFT JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
LEFT JOIN megamind_system_health sh ON r.realm_id = sh.component_name AND sh.component_type = 'realm'
LEFT JOIN megamind_monitoring_alerts ma ON r.realm_id = ma.realm_id AND ma.resolved_at IS NULL
WHERE r.is_active = TRUE
GROUP BY r.realm_id, r.realm_name, r.realm_type, sh.health_score, sh.status
ORDER BY chunk_count DESC;

-- =====================================================================
-- Stored Procedures for Dashboard Data
-- =====================================================================

-- Get dashboard data for realm overview
DELIMITER //
CREATE OR REPLACE PROCEDURE get_realm_dashboard_data(IN p_realm_id VARCHAR(50))
BEGIN
    -- Basic realm information
    SELECT 
        r.realm_id,
        r.realm_name,
        r.realm_type,
        r.description,
        COUNT(c.chunk_id) as total_chunks,
        COUNT(CASE WHEN c.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as chunks_last_week,
        COUNT(CASE WHEN c.last_accessed >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as chunks_accessed_today,
        AVG(c.access_count) as avg_access_count,
        MAX(c.last_accessed) as last_activity,
        COUNT(DISTINCT s.session_id) as active_sessions
    FROM megamind_realms r
    LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
    LEFT JOIN megamind_session_metadata s ON r.realm_id = s.realm_id AND s.is_active = TRUE
    WHERE r.realm_id = p_realm_id AND r.is_active = TRUE
    GROUP BY r.realm_id, r.realm_name, r.realm_type, r.description;
    
    -- Subsystem breakdown
    SELECT 
        COALESCE(ct.tag_value, 'uncategorized') as subsystem,
        COUNT(c.chunk_id) as chunk_count,
        AVG(c.access_count) as avg_access,
        SUM(c.access_count) as total_access
    FROM megamind_chunks c
    LEFT JOIN megamind_chunk_tags ct ON c.chunk_id = ct.chunk_id AND ct.tag_type = 'subsystem'
    WHERE c.realm_id = p_realm_id
    GROUP BY ct.tag_value
    ORDER BY chunk_count DESC
    LIMIT 10;
    
    -- Recent metrics
    SELECT 
        metric_type,
        metric_value,
        period_start,
        period_end
    FROM megamind_realm_metrics
    WHERE realm_id = p_realm_id
    AND measurement_period = 'daily'
    AND period_start >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    ORDER BY period_start DESC
    LIMIT 100;
    
    -- Active alerts
    SELECT 
        alert_id,
        severity,
        title,
        description,
        triggered_at
    FROM megamind_monitoring_alerts
    WHERE realm_id = p_realm_id AND resolved_at IS NULL
    ORDER BY triggered_at DESC
    LIMIT 10;
    
END //
DELIMITER ;

-- Get system overview data
DELIMITER //
CREATE OR REPLACE PROCEDURE get_system_dashboard_data()
BEGIN
    -- System health summary
    SELECT * FROM megamind_system_health_overview;
    
    -- Alert summary
    SELECT * FROM megamind_alert_summary;
    
    -- Realm performance summary
    SELECT * FROM megamind_realm_performance_summary LIMIT 20;
    
    -- System metrics
    SELECT 
        COUNT(DISTINCT r.realm_id) as total_realms,
        COUNT(DISTINCT c.chunk_id) as total_chunks,
        COUNT(DISTINCT cr.relationship_id) as total_relationships,
        COUNT(DISTINCT s.session_id) as active_sessions,
        COUNT(DISTINCT pq.promotion_id) as pending_promotions
    FROM megamind_realms r
    LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
    LEFT JOIN megamind_chunk_relationships cr ON c.chunk_id = cr.chunk_id
    LEFT JOIN megamind_session_metadata s ON r.realm_id = s.realm_id AND s.is_active = TRUE
    LEFT JOIN megamind_promotion_queue pq ON pq.status = 'pending'
    WHERE r.is_active = TRUE;
    
END //
DELIMITER ;

-- =====================================================================
-- Dashboard Configuration Examples
-- =====================================================================

-- Insert sample dashboard configurations
INSERT INTO megamind_dashboards (dashboard_id, name, description, realm_id, layout_config, created_by, is_public) VALUES
('system_overview', 'System Overview', 'Complete system health and performance overview', NULL, 
'{"dashboard_id": "system_overview", "name": "System Overview", "widgets": []}', 'system', TRUE),

('realm_template', 'Realm Dashboard Template', 'Template for realm-specific dashboards', NULL,
'{"dashboard_id": "realm_template", "name": "Realm Dashboard Template", "widgets": []}', 'system', TRUE);

-- =====================================================================
-- Monitoring Event Triggers
-- =====================================================================

-- Trigger to automatically create performance insights based on metrics
DELIMITER //
CREATE OR REPLACE TRIGGER create_performance_insight
AFTER INSERT ON megamind_realm_metrics
FOR EACH ROW
BEGIN
    DECLARE chunk_count INT DEFAULT 0;
    DECLARE avg_access DECIMAL(10,2) DEFAULT 0;
    
    -- Only process daily metrics
    IF NEW.measurement_period = 'daily' THEN
        -- Get current realm statistics
        SELECT COUNT(*), AVG(access_count) INTO chunk_count, avg_access
        FROM megamind_chunks
        WHERE realm_id = NEW.realm_id;
        
        -- Create insight for low activity
        IF NEW.metric_type = 'chunks_accessed' AND NEW.metric_value = 0 AND chunk_count > 0 THEN
            INSERT INTO megamind_performance_insights 
            (insight_id, realm_id, insight_type, severity, title, description, recommendation, impact_score, expires_at)
            VALUES 
            (CONCAT('insight_', REPLACE(UUID(), '-', '')), NEW.realm_id, 'engagement', 'medium',
             'No Chunk Access Today', 
             CONCAT('Realm has ', chunk_count, ' chunks but none were accessed today.'),
             'Review content relevance, improve searchability, or provide user training.',
             0.6, DATE_ADD(NOW(), INTERVAL 7 DAY));
        END IF;
        
        -- Create insight for low engagement
        IF NEW.metric_type = 'chunks_created' AND avg_access < 1.5 AND chunk_count > 10 THEN
            INSERT INTO megamind_performance_insights 
            (insight_id, realm_id, insight_type, severity, title, description, recommendation, impact_score, expires_at)
            VALUES 
            (CONCAT('insight_', REPLACE(UUID(), '-', '')), NEW.realm_id, 'engagement', 'low',
             'Low Content Engagement', 
             CONCAT('Average access count is ', ROUND(avg_access, 2), ' across ', chunk_count, ' chunks.'),
             'Analyze content quality, improve organization, or identify unused content for archival.',
             0.4, DATE_ADD(NOW(), INTERVAL 14 DAY));
        END IF;
    END IF;
END //
DELIMITER ;

-- =====================================================================
-- Cleanup and Maintenance Procedures
-- =====================================================================

-- Procedure to clean up old monitoring data
DELIMITER //
CREATE OR REPLACE PROCEDURE cleanup_monitoring_data()
BEGIN
    DECLARE retention_days INT DEFAULT 90;
    
    -- Clean up resolved alerts older than retention period
    DELETE FROM megamind_monitoring_alerts 
    WHERE resolved_at IS NOT NULL 
    AND resolved_at < DATE_SUB(NOW(), INTERVAL retention_days DAY);
    
    -- Clean up old performance insights
    DELETE FROM megamind_performance_insights 
    WHERE (addressed_at IS NOT NULL AND addressed_at < DATE_SUB(NOW(), INTERVAL retention_days DAY))
    OR (expires_at IS NOT NULL AND expires_at < NOW());
    
    -- Clean up old health check data (keep only latest per component)
    DELETE sh1 FROM megamind_system_health sh1
    JOIN (
        SELECT component_type, component_name, MAX(last_check) as latest_check
        FROM megamind_system_health
        GROUP BY component_type, component_name
    ) sh2 ON sh1.component_type = sh2.component_type 
    AND sh1.component_name = sh2.component_name
    WHERE sh1.last_check < sh2.latest_check;
    
END //
DELIMITER ;

-- Schedule cleanup procedure (requires event scheduler to be enabled)
CREATE EVENT IF NOT EXISTS cleanup_monitoring_data_event
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP + INTERVAL 1 HOUR
DO
CALL cleanup_monitoring_data();

-- =====================================================================
-- Verification and Summary
-- =====================================================================

-- Verify monitoring tables creation
SELECT 
    'Monitoring Tables Created' as status,
    COUNT(*) as table_count
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME IN ('megamind_monitoring_rules', 'megamind_monitoring_alerts', 'megamind_dashboards', 'megamind_performance_insights');

-- Verify monitoring rules insertion
SELECT 
    'Monitoring Rules Configured' as status,
    COUNT(*) as rule_count,
    COUNT(DISTINCT component_type) as component_types
FROM megamind_monitoring_rules;

-- Summary of monitoring setup
SELECT 
    'Phase 4 Monitoring Setup Complete' as message,
    NOW() as setup_timestamp,
    (SELECT COUNT(*) FROM megamind_monitoring_rules) as monitoring_rules,
    (SELECT COUNT(*) FROM megamind_dashboards) as dashboard_configs;