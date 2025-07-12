-- Session Management Tables for Bidirectional Flow
-- Phase 3: Context Database System

-- Drop tables if they exist (for development iterations)
DROP TABLE IF EXISTS megamind_knowledge_contributions;
DROP TABLE IF EXISTS megamind_session_changes;

-- Session Changes Table
-- Stores all pending changes from AI sessions before manual review
CREATE TABLE megamind_session_changes (
    change_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    change_type ENUM('update', 'create', 'relate', 'tag') NOT NULL,
    chunk_id VARCHAR(50),  -- NULL for create operations until commit
    target_chunk_id VARCHAR(50),  -- For relationship operations
    change_data JSON NOT NULL,
    impact_score DECIMAL(3,2) DEFAULT 0.00,  -- 0.00 to 1.00 based on chunk access patterns
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status ENUM('pending', 'approved', 'rejected') DEFAULT 'pending',
    
    INDEX idx_session_changes_session (session_id, timestamp),
    INDEX idx_session_changes_status (status, impact_score DESC),
    INDEX idx_session_changes_chunk (chunk_id, change_type),
    
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- Knowledge Contributions Table
-- Tracks committed AI contributions for analytics and rollback
CREATE TABLE megamind_knowledge_contributions (
    contribution_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    chunks_modified INT DEFAULT 0,
    chunks_created INT DEFAULT 0,
    relationships_added INT DEFAULT 0,
    tags_added INT DEFAULT 0,
    commit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rollback_available BOOLEAN DEFAULT TRUE,
    rollback_data JSON,  -- Stores original state for rollback
    
    INDEX idx_contributions_session (session_id, commit_timestamp),
    INDEX idx_contributions_timestamp (commit_timestamp DESC),
    INDEX idx_contributions_rollback (rollback_available, commit_timestamp)
) ENGINE=InnoDB;

-- Session Metadata Table
-- Tracks active sessions and their context
CREATE TABLE megamind_session_metadata (
    session_id VARCHAR(50) PRIMARY KEY,
    user_context VARCHAR(255),
    project_context VARCHAR(255),
    start_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    pending_changes_count INT DEFAULT 0,
    
    INDEX idx_session_active (is_active, last_activity),
    INDEX idx_session_pending (pending_changes_count DESC, last_activity)
) ENGINE=InnoDB;

-- Create indexes for performance optimization
CREATE INDEX idx_session_changes_impact ON megamind_session_changes (impact_score DESC, timestamp);
CREATE INDEX idx_session_changes_type ON megamind_session_changes (change_type, status);

-- Add sample data for testing
INSERT INTO megamind_session_metadata (session_id, user_context, project_context) VALUES 
('test_session_001', 'development', 'megamind_system'),
('test_session_002', 'analysis', 'spatial_processing');