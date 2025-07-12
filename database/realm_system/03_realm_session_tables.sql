-- Realm-Aware Session Management Tables
-- Phase 1: Session Management with Realm Support
-- Database: megamind_database (MySQL 8.0+)

-- Drop tables if they exist (for development)
DROP TABLE IF EXISTS megamind_knowledge_contributions;
DROP TABLE IF EXISTS megamind_session_changes;
DROP TABLE IF EXISTS megamind_session_metadata;

-- Enhanced session metadata with realm context
CREATE TABLE megamind_session_metadata (
    session_id VARCHAR(50) PRIMARY KEY,
    user_context VARCHAR(255),
    project_context VARCHAR(255),
    start_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    pending_changes_count INT DEFAULT 0,
    
    -- Realm assignment for session
    realm_id VARCHAR(50) NOT NULL,
    
    -- Session configuration
    session_config JSON DEFAULT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE RESTRICT,
    
    -- Indexes
    INDEX idx_session_active (is_active, last_activity),
    INDEX idx_session_pending (pending_changes_count DESC, last_activity),
    INDEX idx_session_realm (realm_id, is_active)
) ENGINE=InnoDB;

-- Enhanced session changes with realm context
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
    
    -- Realm context for change
    source_realm_id VARCHAR(50) NOT NULL,
    target_realm_id VARCHAR(50),  -- For cross-realm operations
    
    -- Foreign key constraints
    FOREIGN KEY (session_id) REFERENCES megamind_session_metadata(session_id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE SET NULL,
    FOREIGN KEY (target_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE SET NULL,
    FOREIGN KEY (source_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE RESTRICT,
    FOREIGN KEY (target_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE RESTRICT,
    
    -- Indexes
    INDEX idx_session_changes_session (session_id, timestamp),
    INDEX idx_session_changes_status (status, impact_score DESC),
    INDEX idx_session_changes_chunk (chunk_id, change_type),
    INDEX idx_session_changes_realm (source_realm_id, status),
    INDEX idx_session_changes_cross_realm (source_realm_id, target_realm_id)
) ENGINE=InnoDB;

-- Enhanced knowledge contributions with realm tracking
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
    
    -- Realm context for contribution
    realm_id VARCHAR(50) NOT NULL,
    
    -- Contribution metadata
    contribution_summary TEXT,
    impact_assessment TEXT,
    
    -- Foreign key constraints
    FOREIGN KEY (session_id) REFERENCES megamind_session_metadata(session_id) ON DELETE RESTRICT,
    FOREIGN KEY (realm_id) REFERENCES megamind_realms(realm_id) ON DELETE RESTRICT,
    
    -- Indexes
    INDEX idx_contributions_session (session_id, commit_timestamp),
    INDEX idx_contributions_timestamp (commit_timestamp DESC),
    INDEX idx_contributions_rollback (rollback_available, commit_timestamp),
    INDEX idx_contributions_realm (realm_id, commit_timestamp DESC)
) ENGINE=InnoDB;