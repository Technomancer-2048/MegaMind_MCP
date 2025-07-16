-- Phase 2: Session Tracking System Schema
-- Enhanced Multi-Embedding Entry System
-- This script adds session management capabilities to the MegaMind database

-- ================================================================
-- SESSION MANAGEMENT TABLES
-- ================================================================

-- Session tracking table
CREATE TABLE IF NOT EXISTS megamind_embedding_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    session_type ENUM('analysis', 'ingestion', 'curation', 'mixed') NOT NULL DEFAULT 'mixed',
    realm_id VARCHAR(64) NOT NULL,
    created_by VARCHAR(255) NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status ENUM('active', 'paused', 'completed', 'cancelled') NOT NULL DEFAULT 'active',
    metadata JSON,
    total_chunks_processed INT DEFAULT 0,
    total_embeddings_generated INT DEFAULT 0,
    processing_duration_ms BIGINT DEFAULT 0,
    
    INDEX idx_session_status (status),
    INDEX idx_session_realm (realm_id),
    INDEX idx_session_activity (last_activity),
    
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Session chunks tracking (links chunks to sessions)
CREATE TABLE IF NOT EXISTS megamind_session_chunks (
    session_chunk_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    chunk_id VARCHAR(64) NOT NULL,
    operation_type ENUM('created', 'updated', 'analyzed', 'embedded', 'quality_assessed') NOT NULL,
    operation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    operation_metadata JSON,
    quality_score DECIMAL(3,2),
    embedding_id VARCHAR(64),
    
    INDEX idx_session_chunk_session (session_id),
    INDEX idx_session_chunk_chunk (chunk_id),
    INDEX idx_session_chunk_operation (operation_type),
    
    FOREIGN KEY (session_id) REFERENCES megamind_embedding_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id),
    FOREIGN KEY (embedding_id) REFERENCES megamind_entry_embeddings(embedding_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Session state tracking
CREATE TABLE IF NOT EXISTS megamind_session_state (
    state_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    state_key VARCHAR(255) NOT NULL,
    state_value JSON NOT NULL,
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_session_state_key (session_id, state_key),
    INDEX idx_session_state_session (session_id),
    
    FOREIGN KEY (session_id) REFERENCES megamind_embedding_sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Session document tracking
CREATE TABLE IF NOT EXISTS megamind_session_documents (
    session_document_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    document_id VARCHAR(64) NOT NULL,
    document_path TEXT,
    document_hash VARCHAR(64),
    processing_status ENUM('pending', 'processing', 'completed', 'failed') NOT NULL DEFAULT 'pending',
    chunks_created INT DEFAULT 0,
    embeddings_created INT DEFAULT 0,
    quality_scores JSON,
    error_details TEXT,
    started_at TIMESTAMP NULL,
    completed_at TIMESTAMP NULL,
    
    INDEX idx_session_doc_session (session_id),
    INDEX idx_session_doc_status (processing_status),
    
    FOREIGN KEY (session_id) REFERENCES megamind_embedding_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES megamind_document_structures(document_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Session metrics tracking
CREATE TABLE IF NOT EXISTS megamind_session_metrics (
    metric_id VARCHAR(64) PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    metric_type VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,4) NOT NULL,
    metric_unit VARCHAR(50),
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    
    INDEX idx_session_metrics_session (session_id),
    INDEX idx_session_metrics_type (metric_type),
    INDEX idx_session_metrics_time (recorded_at),
    
    FOREIGN KEY (session_id) REFERENCES megamind_embedding_sessions(session_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ================================================================
-- ENHANCED TABLES FOR SESSION AWARENESS
-- ================================================================

-- Add session tracking to document structures
ALTER TABLE megamind_document_structures
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS session_metadata JSON,
ADD INDEX idx_document_session (session_id),
ADD FOREIGN KEY fk_document_session (session_id) 
    REFERENCES megamind_embedding_sessions(session_id) ON DELETE SET NULL;

-- Add session tracking to chunk metadata
ALTER TABLE megamind_chunk_metadata
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS session_operation VARCHAR(50),
ADD INDEX idx_chunk_meta_session (session_id),
ADD FOREIGN KEY fk_chunk_meta_session (session_id) 
    REFERENCES megamind_embedding_sessions(session_id) ON DELETE SET NULL;

-- Add session tracking to embeddings
ALTER TABLE megamind_entry_embeddings
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS session_metadata JSON,
ADD INDEX idx_embedding_session (session_id),
ADD FOREIGN KEY fk_embedding_session (session_id) 
    REFERENCES megamind_embedding_sessions(session_id) ON DELETE SET NULL;

-- Add session tracking to quality assessments
ALTER TABLE megamind_quality_assessments
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD INDEX idx_quality_session (session_id),
ADD FOREIGN KEY fk_quality_session (session_id) 
    REFERENCES megamind_embedding_sessions(session_id) ON DELETE SET NULL;

-- ================================================================
-- VIEWS FOR SESSION OPERATIONS
-- ================================================================

-- Active sessions view
CREATE OR REPLACE VIEW megamind_active_sessions_view AS
SELECT 
    s.session_id,
    s.session_type,
    s.realm_id,
    r.realm_name,
    s.created_by,
    s.created_date,
    s.last_activity,
    s.status,
    s.total_chunks_processed,
    s.total_embeddings_generated,
    s.processing_duration_ms,
    COUNT(DISTINCT sd.document_id) as document_count,
    COUNT(DISTINCT sc.chunk_id) as unique_chunks,
    AVG(sc.quality_score) as avg_quality_score
FROM megamind_embedding_sessions s
JOIN megamind_realms r ON s.realm_id = r.realm_id
LEFT JOIN megamind_session_documents sd ON s.session_id = sd.session_id
LEFT JOIN megamind_session_chunks sc ON s.session_id = sc.session_id
WHERE s.status = 'active'
GROUP BY s.session_id;

-- Session progress view
CREATE OR REPLACE VIEW megamind_session_progress_view AS
SELECT 
    s.session_id,
    s.session_type,
    s.status,
    COUNT(DISTINCT CASE WHEN sd.processing_status = 'completed' THEN sd.document_id END) as completed_documents,
    COUNT(DISTINCT CASE WHEN sd.processing_status = 'processing' THEN sd.document_id END) as processing_documents,
    COUNT(DISTINCT CASE WHEN sd.processing_status = 'pending' THEN sd.document_id END) as pending_documents,
    COUNT(DISTINCT CASE WHEN sd.processing_status = 'failed' THEN sd.document_id END) as failed_documents,
    SUM(sd.chunks_created) as total_chunks_created,
    SUM(sd.embeddings_created) as total_embeddings_created,
    AVG(TIMESTAMPDIFF(SECOND, sd.started_at, sd.completed_at)) as avg_document_processing_time
FROM megamind_embedding_sessions s
LEFT JOIN megamind_session_documents sd ON s.session_id = sd.session_id
GROUP BY s.session_id;

-- ================================================================
-- STORED PROCEDURES FOR SESSION MANAGEMENT
-- ================================================================

DELIMITER //

-- Create new session
CREATE PROCEDURE IF NOT EXISTS sp_create_embedding_session(
    IN p_session_type VARCHAR(50),
    IN p_realm_id VARCHAR(64),
    IN p_created_by VARCHAR(255),
    IN p_metadata JSON,
    OUT p_session_id VARCHAR(64)
)
BEGIN
    SET p_session_id = CONCAT('session_', MD5(CONCAT(UUID(), NOW())));
    
    INSERT INTO megamind_embedding_sessions (
        session_id, session_type, realm_id, created_by, metadata
    ) VALUES (
        p_session_id, p_session_type, p_realm_id, p_created_by, p_metadata
    );
END//

-- Track chunk operation in session
CREATE PROCEDURE IF NOT EXISTS sp_track_session_chunk_operation(
    IN p_session_id VARCHAR(64),
    IN p_chunk_id VARCHAR(64),
    IN p_operation_type VARCHAR(50),
    IN p_operation_metadata JSON,
    IN p_quality_score DECIMAL(3,2),
    IN p_embedding_id VARCHAR(64)
)
BEGIN
    DECLARE v_session_chunk_id VARCHAR(64);
    SET v_session_chunk_id = CONCAT('sc_', MD5(CONCAT(p_session_id, p_chunk_id, NOW())));
    
    INSERT INTO megamind_session_chunks (
        session_chunk_id, session_id, chunk_id, operation_type, 
        operation_metadata, quality_score, embedding_id
    ) VALUES (
        v_session_chunk_id, p_session_id, p_chunk_id, p_operation_type,
        p_operation_metadata, p_quality_score, p_embedding_id
    );
    
    -- Update session metrics
    UPDATE megamind_embedding_sessions 
    SET 
        total_chunks_processed = total_chunks_processed + 1,
        total_embeddings_generated = total_embeddings_generated + IF(p_embedding_id IS NOT NULL, 1, 0)
    WHERE session_id = p_session_id;
END//

-- Update session state
CREATE PROCEDURE IF NOT EXISTS sp_update_session_state(
    IN p_session_id VARCHAR(64),
    IN p_state_key VARCHAR(255),
    IN p_state_value JSON
)
BEGIN
    INSERT INTO megamind_session_state (
        state_id, session_id, state_key, state_value
    ) VALUES (
        CONCAT('state_', MD5(CONCAT(p_session_id, p_state_key, NOW()))),
        p_session_id, p_state_key, p_state_value
    )
    ON DUPLICATE KEY UPDATE 
        state_value = p_state_value,
        updated_timestamp = CURRENT_TIMESTAMP;
END//

-- Complete session
CREATE PROCEDURE IF NOT EXISTS sp_complete_session(
    IN p_session_id VARCHAR(64)
)
BEGIN
    UPDATE megamind_embedding_sessions 
    SET 
        status = 'completed',
        processing_duration_ms = TIMESTAMPDIFF(MICROSECOND, created_date, NOW()) / 1000
    WHERE session_id = p_session_id;
END//

DELIMITER ;

-- ================================================================
-- INDEXES FOR PERFORMANCE
-- ================================================================

-- Session performance indexes
CREATE INDEX IF NOT EXISTS idx_session_realm_status 
    ON megamind_embedding_sessions(realm_id, status);

CREATE INDEX IF NOT EXISTS idx_session_chunk_timestamp 
    ON megamind_session_chunks(session_id, operation_timestamp);

CREATE INDEX IF NOT EXISTS idx_session_doc_completed 
    ON megamind_session_documents(session_id, completed_at);

-- ================================================================
-- INITIAL CONFIGURATION
-- ================================================================

-- Add session configuration to system config
INSERT INTO megamind_system_config (config_key, config_value, description) VALUES
('session.default_timeout_minutes', '120', 'Default session timeout in minutes'),
('session.max_concurrent_sessions', '10', 'Maximum concurrent active sessions per realm'),
('session.chunk_batch_size', '100', 'Default batch size for chunk processing'),
('session.embedding_batch_size', '32', 'Default batch size for embedding generation'),
('session.quality_threshold', '0.7', 'Minimum quality score threshold'),
('session.auto_save_interval_seconds', '300', 'Auto-save session state interval')
ON DUPLICATE KEY UPDATE 
    config_value = VALUES(config_value),
    updated_date = CURRENT_TIMESTAMP;

-- ================================================================
-- GRANTS (if needed)
-- ================================================================

-- Grant necessary permissions to the application user
-- GRANT ALL PRIVILEGES ON megamind_embedding_sessions TO 'megamind_app'@'%';
-- GRANT ALL PRIVILEGES ON megamind_session_chunks TO 'megamind_app'@'%';
-- GRANT ALL PRIVILEGES ON megamind_session_state TO 'megamind_app'@'%';
-- GRANT ALL PRIVILEGES ON megamind_session_documents TO 'megamind_app'@'%';
-- GRANT ALL PRIVILEGES ON megamind_session_metrics TO 'megamind_app'@'%';

-- Schema update complete
SELECT 'Phase 2 Session Tracking System schema created successfully' as status;