-- Context Database System - Session Management Tables
-- Phase 3: Bidirectional Flow Support

-- Session changes table for buffering modifications
CREATE TABLE megamind_session_changes (
    change_id INT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(50) NOT NULL,
    change_type ENUM('update', 'create', 'relate', 'tag', 'delete') NOT NULL,
    chunk_id VARCHAR(50),
    change_data JSON NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('pending', 'approved', 'rejected', 'committed') NOT NULL DEFAULT 'pending',
    
    -- Reference to original chunk (nullable for creates)
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    
    -- Index for session-based queries
    INDEX idx_session_changes_session (session_id, timestamp),
    INDEX idx_session_changes_status (status, timestamp),
    INDEX idx_session_changes_chunk (chunk_id, change_type)
);

-- Knowledge contributions tracking
CREATE TABLE megamind_knowledge_contributions (
    contribution_id INT PRIMARY KEY AUTO_INCREMENT,
    session_id VARCHAR(50) NOT NULL,
    chunks_modified INT DEFAULT 0,
    chunks_created INT DEFAULT 0,
    relationships_added INT DEFAULT 0,
    tags_added INT DEFAULT 0,
    commit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rollback_available BOOLEAN DEFAULT TRUE,
    
    -- Summary statistics
    total_tokens_added INT DEFAULT 0,
    total_tokens_modified INT DEFAULT 0,
    impact_score DECIMAL(5,2) DEFAULT 0.0,
    
    -- Metadata
    contributor_info JSON DEFAULT NULL,
    review_notes TEXT DEFAULT NULL,
    
    INDEX idx_contributions_session (session_id, commit_timestamp),
    INDEX idx_contributions_impact (impact_score DESC, commit_timestamp)
);

-- Session metadata table
CREATE TABLE megamind_sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status ENUM('active', 'review_pending', 'committed', 'abandoned') NOT NULL DEFAULT 'active',
    
    -- Session context
    project_context JSON DEFAULT NULL,
    goals TEXT DEFAULT NULL,
    
    -- Statistics
    queries_executed INT DEFAULT 0,
    chunks_accessed INT DEFAULT 0,
    changes_pending INT DEFAULT 0,
    
    INDEX idx_sessions_status (status, last_activity),
    INDEX idx_sessions_activity (last_activity DESC)
);