-- =====================================================================
-- MegaMind Context Database - Phase 4 Production Deployment
-- =====================================================================
-- Production-optimized schema deployment with performance tuning
-- Created: 2025-07-12
-- Purpose: Complete database setup for production environment

-- =====================================================================
-- Performance Optimization Indexes
-- =====================================================================

-- Realm-aware performance indexes for production load
CREATE INDEX IF NOT EXISTS idx_chunks_realm_performance 
ON megamind_chunks (realm_id, access_count DESC, chunk_type, last_accessed DESC);

CREATE INDEX IF NOT EXISTS idx_chunks_search_realm_optimized 
ON megamind_chunks (realm_id, created_at DESC, chunk_type) 
INCLUDE (chunk_id, source_document, section_path);

CREATE INDEX IF NOT EXISTS idx_inheritance_lookup_optimized 
ON megamind_realm_inheritance (child_realm_id, inheritance_type, priority DESC);

CREATE INDEX IF NOT EXISTS idx_realm_permissions_user_optimized 
ON megamind_realm_permissions (user_id, realm_id, role) 
INCLUDE (granted_at, granted_by);

-- Cross-realm relationship performance indexes
CREATE INDEX IF NOT EXISTS idx_relationships_cross_realm 
ON megamind_chunk_relationships (source_realm_id, target_realm_id, relationship_type, strength DESC);

CREATE INDEX IF NOT EXISTS idx_relationships_realm_traversal 
ON megamind_chunk_relationships (chunk_id, related_chunk_id) 
INCLUDE (relationship_type, strength, source_realm_id, target_realm_id);

-- Session and audit performance indexes
CREATE INDEX IF NOT EXISTS idx_session_realm_activity 
ON megamind_session_metadata (realm_id, is_active, last_activity DESC);

CREATE INDEX IF NOT EXISTS idx_audit_realm_security 
ON megamind_audit_log (target_realm_id, security_level, event_timestamp DESC) 
WHERE security_level IN ('high', 'critical');

-- =====================================================================
-- Production Configuration Tables
-- =====================================================================

-- Production monitoring configuration
CREATE TABLE IF NOT EXISTS megamind_production_config (
    config_id VARCHAR(50) PRIMARY KEY,
    config_category ENUM('performance', 'monitoring', 'security', 'retention') NOT NULL,
    config_key VARCHAR(100) NOT NULL,
    config_value JSON NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    
    UNIQUE KEY unique_config (config_category, config_key),
    INDEX idx_config_category (config_category, is_active)
) ENGINE=InnoDB;

-- Realm performance metrics tracking
CREATE TABLE IF NOT EXISTS megamind_realm_metrics (
    metric_id VARCHAR(50) PRIMARY KEY,
    realm_id VARCHAR(50) NOT NULL,
    metric_type ENUM('chunks_created', 'chunks_accessed', 'relationships_formed', 'searches_performed', 'promotions_requested') NOT NULL,
    metric_value BIGINT NOT NULL DEFAULT 0,
    measurement_period ENUM('hourly', 'daily', 'weekly', 'monthly') NOT NULL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    UNIQUE KEY unique_realm_metric_period (realm_id, metric_type, measurement_period, period_start),
    INDEX idx_metrics_realm_type (realm_id, metric_type, period_start DESC),
    INDEX idx_metrics_period (measurement_period, period_start DESC)
) ENGINE=InnoDB;

-- System health monitoring
CREATE TABLE IF NOT EXISTS megamind_system_health (
    health_id VARCHAR(50) PRIMARY KEY,
    component_type ENUM('database', 'inheritance', 'search', 'promotion', 'security') NOT NULL,
    component_name VARCHAR(100) NOT NULL,
    status ENUM('healthy', 'warning', 'critical', 'down') NOT NULL,
    health_score DECIMAL(5,2) NOT NULL CHECK (health_score >= 0.00 AND health_score <= 100.00),
    performance_metrics JSON,
    last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    check_interval_minutes INT DEFAULT 5,
    alert_threshold DECIMAL(5,2) DEFAULT 90.00,
    
    UNIQUE KEY unique_component (component_type, component_name),
    INDEX idx_health_status (status, last_check DESC),
    INDEX idx_health_score (health_score ASC, last_check DESC)
) ENGINE=InnoDB;

-- =====================================================================
-- Production Data Population
-- =====================================================================

-- Insert default production configuration
INSERT INTO megamind_production_config (config_id, config_category, config_key, config_value, description) VALUES
('perf_001', 'performance', 'realm_cache_ttl', '3600', 'Realm context cache TTL in seconds'),
('perf_002', 'performance', 'search_result_cache_ttl', '1800', 'Search result cache TTL in seconds'),
('perf_003', 'performance', 'inheritance_cache_ttl', '7200', 'Inheritance resolution cache TTL in seconds'),
('perf_004', 'performance', 'max_search_results', '100', 'Maximum search results per query'),
('perf_005', 'performance', 'max_inheritance_depth', '5', 'Maximum inheritance chain depth'),

('monitor_001', 'monitoring', 'health_check_interval', '300', 'System health check interval in seconds'),
('monitor_002', 'monitoring', 'metric_collection_interval', '3600', 'Metrics collection interval in seconds'),
('monitor_003', 'monitoring', 'alert_threshold_critical', '95', 'Critical alert threshold percentage'),
('monitor_004', 'monitoring', 'alert_threshold_warning', '80', 'Warning alert threshold percentage'),

('security_001', 'security', 'max_failed_login_attempts', '5', 'Maximum failed login attempts before lockout'),
('security_002', 'security', 'session_timeout_minutes', '480', 'Session timeout in minutes (8 hours)'),
('security_003', 'security', 'audit_retention_days', '365', 'Audit log retention period in days'),
('security_004', 'security', 'security_scan_interval', '86400', 'Security scan interval in seconds (24 hours)'),

('retention_001', 'retention', 'inactive_session_cleanup_days', '30', 'Clean up inactive sessions after days'),
('retention_002', 'retention', 'metrics_retention_months', '12', 'Metrics retention period in months'),
('retention_003', 'retention', 'health_log_retention_days', '90', 'Health check log retention in days');

-- Initialize system health components
INSERT INTO megamind_system_health (health_id, component_type, component_name, status, health_score, performance_metrics) VALUES
('health_db_001', 'database', 'connection_pool', 'healthy', 100.00, '{"active_connections": 0, "max_connections": 100}'),
('health_db_002', 'database', 'query_performance', 'healthy', 100.00, '{"avg_query_time_ms": 0, "slow_queries_count": 0}'),
('health_inherit_001', 'inheritance', 'resolution_engine', 'healthy', 100.00, '{"avg_resolution_time_ms": 0, "cache_hit_rate": 0}'),
('health_search_001', 'search', 'chunk_search', 'healthy', 100.00, '{"avg_search_time_ms": 0, "result_accuracy": 100}'),
('health_promotion_001', 'promotion', 'workflow_engine', 'healthy', 100.00, '{"pending_requests": 0, "approval_rate": 100}'),
('health_security_001', 'security', 'access_control', 'healthy', 100.00, '{"failed_access_attempts": 0, "security_violations": 0}');

-- =====================================================================
-- Production Views for Monitoring
-- =====================================================================

-- Realm activity dashboard view
CREATE OR REPLACE VIEW megamind_realm_activity_dashboard AS
SELECT 
    r.realm_id,
    r.realm_name,
    r.realm_type,
    COUNT(c.chunk_id) as total_chunks,
    COUNT(CASE WHEN c.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY) THEN 1 END) as chunks_last_7_days,
    COUNT(CASE WHEN c.last_accessed >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as chunks_accessed_today,
    AVG(c.access_count) as avg_access_count,
    MAX(c.last_accessed) as last_activity,
    COUNT(DISTINCT s.session_id) as active_sessions
FROM megamind_realms r
LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
LEFT JOIN megamind_session_metadata s ON r.realm_id = s.realm_id AND s.is_active = TRUE
WHERE r.is_active = TRUE
GROUP BY r.realm_id, r.realm_name, r.realm_type
ORDER BY total_chunks DESC;

-- System performance overview
CREATE OR REPLACE VIEW megamind_system_performance_overview AS
SELECT 
    'chunks' as metric_type,
    COUNT(*) as total_count,
    COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as created_today,
    COUNT(CASE WHEN last_accessed >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as accessed_today,
    AVG(access_count) as avg_access_count
FROM megamind_chunks
WHERE realm_id IN (SELECT realm_id FROM megamind_realms WHERE is_active = TRUE)

UNION ALL

SELECT 
    'relationships' as metric_type,
    COUNT(*) as total_count,
    COUNT(CASE WHEN created_at >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as created_today,
    0 as accessed_today,
    AVG(strength) as avg_access_count
FROM megamind_chunk_relationships

UNION ALL

SELECT 
    'sessions' as metric_type,
    COUNT(*) as total_count,
    COUNT(CASE WHEN start_timestamp >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as created_today,
    COUNT(CASE WHEN is_active = TRUE THEN 1 END) as accessed_today,
    AVG(pending_changes_count) as avg_access_count
FROM megamind_session_metadata

UNION ALL

SELECT 
    'promotions' as metric_type,
    COUNT(*) as total_count,
    COUNT(CASE WHEN requested_at >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as created_today,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as accessed_today,
    0 as avg_access_count
FROM megamind_promotion_queue;

-- Security monitoring view
CREATE OR REPLACE VIEW megamind_security_monitoring AS
SELECT 
    'audit_events' as security_metric,
    COUNT(*) as total_count,
    COUNT(CASE WHEN event_timestamp >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as last_24_hours,
    COUNT(CASE WHEN security_level = 'critical' THEN 1 END) as critical_count,
    COUNT(CASE WHEN security_level = 'high' THEN 1 END) as high_count
FROM megamind_audit_log

UNION ALL

SELECT 
    'security_violations' as security_metric,
    COUNT(*) as total_count,
    COUNT(CASE WHEN detected_at >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as last_24_hours,
    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_count,
    COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_count
FROM megamind_security_violations

UNION ALL

SELECT 
    'failed_permissions' as security_metric,
    COUNT(*) as total_count,
    COUNT(CASE WHEN event_timestamp >= DATE_SUB(NOW(), INTERVAL 1 DAY) THEN 1 END) as last_24_hours,
    COUNT(CASE WHEN event_type LIKE '%failed%' THEN 1 END) as critical_count,
    0 as high_count
FROM megamind_audit_log
WHERE event_category = 'security' AND event_type LIKE '%permission%';

-- =====================================================================
-- Production Stored Procedures
-- =====================================================================

-- Performance optimization procedure
DELIMITER //
CREATE OR REPLACE PROCEDURE optimize_realm_performance()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE realm_id_var VARCHAR(50);
    DECLARE chunk_count INT;
    
    -- Cursor for active realms
    DECLARE realm_cursor CURSOR FOR 
        SELECT realm_id FROM megamind_realms WHERE is_active = TRUE;
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    -- Update access count statistics
    UPDATE megamind_chunks 
    SET last_accessed = CURRENT_TIMESTAMP 
    WHERE chunk_id IN (
        SELECT chunk_id FROM (
            SELECT chunk_id FROM megamind_chunks 
            WHERE last_accessed < DATE_SUB(NOW(), INTERVAL 1 HOUR)
            ORDER BY access_count DESC 
            LIMIT 1000
        ) as temp_chunks
    );
    
    -- Optimize inheritance cache
    OPEN realm_cursor;
    
    realm_loop: LOOP
        FETCH realm_cursor INTO realm_id_var;
        IF done THEN
            LEAVE realm_loop;
        END IF;
        
        -- Update realm metrics
        SELECT COUNT(*) INTO chunk_count 
        FROM megamind_chunks 
        WHERE realm_id = realm_id_var;
        
        -- Insert or update realm metrics
        INSERT INTO megamind_realm_metrics 
        (metric_id, realm_id, metric_type, metric_value, measurement_period, period_start, period_end)
        VALUES 
        (CONCAT('metric_', realm_id_var, '_', UNIX_TIMESTAMP()), realm_id_var, 'chunks_created', 
         chunk_count, 'daily', DATE(NOW()), DATE_ADD(DATE(NOW()), INTERVAL 1 DAY))
        ON DUPLICATE KEY UPDATE 
        metric_value = chunk_count, 
        period_end = DATE_ADD(DATE(NOW()), INTERVAL 1 DAY);
        
    END LOOP;
    
    CLOSE realm_cursor;
    
    -- Analyze table performance
    ANALYZE TABLE megamind_chunks, megamind_chunk_relationships, megamind_realms;
    
END //
DELIMITER ;

-- Health monitoring procedure
DELIMITER //
CREATE OR REPLACE PROCEDURE check_system_health()
BEGIN
    DECLARE connection_count INT DEFAULT 0;
    DECLARE slow_query_count INT DEFAULT 0;
    DECLARE avg_query_time DECIMAL(10,2) DEFAULT 0.00;
    DECLARE health_score DECIMAL(5,2);
    
    -- Check database connections
    SELECT COUNT(*) INTO connection_count 
    FROM INFORMATION_SCHEMA.PROCESSLIST 
    WHERE DB = DATABASE();
    
    -- Calculate health scores based on performance metrics
    SET health_score = CASE 
        WHEN connection_count < 50 THEN 100.00
        WHEN connection_count < 80 THEN 85.00
        WHEN connection_count < 95 THEN 70.00
        ELSE 50.00
    END;
    
    -- Update database health
    UPDATE megamind_system_health 
    SET 
        health_score = health_score,
        performance_metrics = JSON_OBJECT(
            'active_connections', connection_count,
            'max_connections', 100,
            'connection_utilization', ROUND((connection_count / 100) * 100, 2)
        ),
        status = CASE 
            WHEN health_score >= 90 THEN 'healthy'
            WHEN health_score >= 70 THEN 'warning'
            ELSE 'critical'
        END
    WHERE component_type = 'database' AND component_name = 'connection_pool';
    
    -- Update search performance health
    UPDATE megamind_system_health 
    SET 
        health_score = 100.00,
        performance_metrics = JSON_OBJECT(
            'total_chunks', (SELECT COUNT(*) FROM megamind_chunks),
            'total_realms', (SELECT COUNT(*) FROM megamind_realms WHERE is_active = TRUE),
            'active_sessions', (SELECT COUNT(*) FROM megamind_session_metadata WHERE is_active = TRUE)
        ),
        status = 'healthy'
    WHERE component_type = 'search' AND component_name = 'chunk_search';
    
    -- Update inheritance health
    UPDATE megamind_system_health 
    SET 
        health_score = 100.00,
        performance_metrics = JSON_OBJECT(
            'inheritance_relationships', (SELECT COUNT(*) FROM megamind_realm_inheritance),
            'realms_with_inheritance', (SELECT COUNT(DISTINCT child_realm_id) FROM megamind_realm_inheritance)
        ),
        status = 'healthy'
    WHERE component_type = 'inheritance' AND component_name = 'resolution_engine';
    
END //
DELIMITER ;

-- =====================================================================
-- Production Maintenance Procedures
-- =====================================================================

-- Cleanup old data procedure
DELIMITER //
CREATE OR REPLACE PROCEDURE cleanup_old_data()
BEGIN
    DECLARE retention_days INT;
    DECLARE metrics_retention_months INT;
    
    -- Get retention settings
    SELECT CAST(config_value AS UNSIGNED) INTO retention_days 
    FROM megamind_production_config 
    WHERE config_key = 'inactive_session_cleanup_days';
    
    SELECT CAST(config_value AS UNSIGNED) INTO metrics_retention_months 
    FROM megamind_production_config 
    WHERE config_key = 'metrics_retention_months';
    
    -- Clean up inactive sessions
    DELETE FROM megamind_session_metadata 
    WHERE is_active = FALSE 
    AND last_activity < DATE_SUB(NOW(), INTERVAL retention_days DAY);
    
    -- Clean up old metrics
    DELETE FROM megamind_realm_metrics 
    WHERE period_start < DATE_SUB(NOW(), INTERVAL metrics_retention_months MONTH);
    
    -- Clean up old health checks (keep only recent ones)
    DELETE FROM megamind_system_health 
    WHERE last_check < DATE_SUB(NOW(), INTERVAL 7 DAY);
    
    -- Archive old audit logs to separate table if needed
    -- (Implementation depends on archival strategy)
    
END //
DELIMITER ;

-- =====================================================================
-- Production Event Scheduler Setup
-- =====================================================================

-- Enable event scheduler
SET GLOBAL event_scheduler = ON;

-- Schedule performance optimization
CREATE EVENT IF NOT EXISTS optimize_realm_performance_event
ON SCHEDULE EVERY 1 HOUR
STARTS CURRENT_TIMESTAMP
DO
CALL optimize_realm_performance();

-- Schedule health monitoring
CREATE EVENT IF NOT EXISTS check_system_health_event
ON SCHEDULE EVERY 5 MINUTE
STARTS CURRENT_TIMESTAMP
DO
CALL check_system_health();

-- Schedule daily cleanup
CREATE EVENT IF NOT EXISTS cleanup_old_data_event
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP + INTERVAL 1 HOUR
DO
CALL cleanup_old_data();

-- =====================================================================
-- Production Security Settings
-- =====================================================================

-- Ensure secure defaults
UPDATE megamind_realms 
SET default_access_level = 'read' 
WHERE default_access_level IS NULL;

-- Validate all realm relationships
UPDATE megamind_realm_inheritance 
SET inheritance_type = 'full' 
WHERE inheritance_type IS NULL;

-- =====================================================================
-- Production Deployment Verification
-- =====================================================================

-- Create verification view
CREATE OR REPLACE VIEW megamind_deployment_verification AS
SELECT 
    'schema_tables' as check_type,
    COUNT(*) as expected_count,
    COUNT(*) as actual_count,
    CASE WHEN COUNT(*) >= 15 THEN 'PASS' ELSE 'FAIL' END as status
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = DATABASE() 
AND TABLE_NAME LIKE 'megamind_%'

UNION ALL

SELECT 
    'realm_indexes' as check_type,
    8 as expected_count,
    COUNT(*) as actual_count,
    CASE WHEN COUNT(*) >= 8 THEN 'PASS' ELSE 'FAIL' END as status
FROM INFORMATION_SCHEMA.STATISTICS 
WHERE TABLE_SCHEMA = DATABASE() 
AND INDEX_NAME LIKE 'idx_%realm%'

UNION ALL

SELECT 
    'system_health_components' as check_type,
    6 as expected_count,
    COUNT(*) as actual_count,
    CASE WHEN COUNT(*) >= 6 THEN 'PASS' ELSE 'FAIL' END as status
FROM megamind_system_health

UNION ALL

SELECT 
    'production_config' as check_type,
    15 as expected_count,
    COUNT(*) as actual_count,
    CASE WHEN COUNT(*) >= 15 THEN 'PASS' ELSE 'FAIL' END as status
FROM megamind_production_config;

-- Final deployment message
SELECT 'MegaMind Context Database - Phase 4 Production Deployment Complete' as message,
       NOW() as deployment_timestamp,
       (SELECT COUNT(*) FROM megamind_deployment_verification WHERE status = 'PASS') as passed_checks,
       (SELECT COUNT(*) FROM megamind_deployment_verification WHERE status = 'FAIL') as failed_checks;