-- MegaMind Context Database Schema
-- Creates all required tables for the complete system

USE megamind_database;

-- Phase 1: Core Infrastructure Tables
CREATE TABLE IF NOT EXISTS megamind_chunks (
    chunk_id VARCHAR(50) PRIMARY KEY,
    realm_id VARCHAR(50) NOT NULL DEFAULT 'GLOBAL',
    content TEXT NOT NULL,
    embedding JSON NULL,
    complexity_score FLOAT DEFAULT 0.0,
    source_document VARCHAR(255) NOT NULL,
    section_path VARCHAR(500),
    chunk_type ENUM('rule', 'function', 'section', 'example') DEFAULT 'section',
    line_count INT DEFAULT 0,
    token_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INT DEFAULT 0,
    INDEX idx_source_doc (source_document),
    INDEX idx_chunk_type (chunk_type),
    INDEX idx_access_count (access_count DESC),
    INDEX idx_chunks_realm (realm_id),
    FULLTEXT(content, section_path)
);

CREATE TABLE IF NOT EXISTS megamind_chunk_relationships (
    relationship_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    related_chunk_id VARCHAR(50) NOT NULL,
    relationship_type ENUM('references', 'implements', 'extends', 'uses', 'similar_to') NOT NULL,
    strength DECIMAL(3,2) DEFAULT 0.0,
    discovered_by ENUM('manual', 'semantic', 'structural') DEFAULT 'manual',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (related_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    INDEX idx_chunk_relationships (chunk_id),
    INDEX idx_relationship_type (relationship_type)
);

CREATE TABLE IF NOT EXISTS megamind_chunk_tags (
    tag_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    tag_type ENUM('keyword', 'category', 'priority', 'status', 'language') NOT NULL,
    tag_value VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    INDEX idx_chunk_tags (chunk_id),
    INDEX idx_tag_type_value (tag_type, tag_value)
);

-- Phase 2: Intelligence Layer Tables
CREATE TABLE IF NOT EXISTS megamind_embeddings (
    embedding_id VARCHAR(50) PRIMARY KEY,
    realm_id VARCHAR(50) NOT NULL DEFAULT 'GLOBAL',
    chunk_id VARCHAR(50) NOT NULL,
    embedding_vector JSON NOT NULL,
    model_name VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    INDEX idx_embedding_chunk (chunk_id),
    INDEX idx_embeddings_realm (realm_id)
);

-- Phase 3: Session Management Tables
CREATE TABLE IF NOT EXISTS megamind_session_changes (
    change_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    change_type ENUM('create_chunk', 'update_chunk', 'add_relationship', 'add_tag') NOT NULL,
    target_chunk_id VARCHAR(50),
    change_data JSON NOT NULL,
    impact_score DECIMAL(3,2) DEFAULT 0.0,
    priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    status ENUM('pending', 'approved', 'rejected', 'applied') DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_session_changes (session_id),
    INDEX idx_change_status (status),
    INDEX idx_impact_score (impact_score DESC)
);

CREATE TABLE IF NOT EXISTS megamind_session_metadata (
    session_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(100),
    project_context VARCHAR(255),
    session_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_sessions (user_id),
    INDEX idx_last_activity (last_activity DESC)
);

CREATE TABLE IF NOT EXISTS megamind_knowledge_contributions (
    contribution_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    contributor_type ENUM('human', 'ai_claude', 'ai_gpt', 'automated') DEFAULT 'ai_claude',
    contribution_summary TEXT,
    chunks_affected INT DEFAULT 0,
    chunks_modified INT DEFAULT 0,
    chunks_created INT DEFAULT 0,
    relationships_added INT DEFAULT 0,
    contribution_impact INT DEFAULT 0,
    quality_score DECIMAL(3,2) DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES megamind_session_metadata(session_id) ON DELETE CASCADE,
    INDEX idx_contributor_type (contributor_type),
    INDEX idx_quality_score (quality_score DESC),
    INDEX idx_chunks_modified (chunks_modified DESC),
    INDEX idx_chunks_created (chunks_created DESC),
    INDEX idx_contribution_impact (contribution_impact DESC)
);

-- Phase 4: Advanced Optimization Tables
CREATE TABLE IF NOT EXISTS megamind_performance_metrics (
    metric_id VARCHAR(50) PRIMARY KEY,
    metric_type ENUM('query_time', 'chunk_access', 'session_length', 'error_rate', 'access_tracking') NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    chunk_id VARCHAR(50),
    realm_id VARCHAR(50),
    query_context TEXT,
    context_data JSON,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_metric_type_time (metric_type, recorded_at DESC),
    INDEX idx_chunk_realm (chunk_id, realm_id)
);

CREATE TABLE IF NOT EXISTS megamind_system_health (
    health_id VARCHAR(50) PRIMARY KEY,
    component ENUM('database', 'mcp_server', 'analytics', 'review_interface') NOT NULL,
    status ENUM('healthy', 'warning', 'critical', 'down') NOT NULL,
    details JSON,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_component_status (component, status),
    INDEX idx_checked_at (checked_at DESC)
);

-- Insert sample data for testing (with access_count = 1 since creation counts as first access)
INSERT IGNORE INTO megamind_chunks (chunk_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count) VALUES
('sample_001', 'This is a sample chunk for testing the MegaMind Context Database System. It demonstrates basic functionality and search capabilities.', 'sample_document.md', '/Introduction/Overview', 'section', 2, 25, 1),
('sample_002', 'def search_function(query):\n    return database.search(query)', 'sample_code.py', '/Functions/SearchFunction', 'function', 2, 15, 1),
('sample_003', 'Always use transactions when modifying database records to ensure data consistency and integrity.', 'database_rules.md', '/Rules/Transactions', 'rule', 1, 18, 1);

-- Insert sample relationships
INSERT IGNORE INTO megamind_chunk_relationships (relationship_id, chunk_id, related_chunk_id, relationship_type, strength) VALUES
('rel_001', 'sample_001', 'sample_002', 'references', 0.8),
('rel_002', 'sample_002', 'sample_003', 'implements', 0.9);

-- Insert sample tags
INSERT IGNORE INTO megamind_chunk_tags (tag_id, chunk_id, tag_type, tag_value) VALUES
('tag_001', 'sample_001', 'category', 'introduction'),
('tag_002', 'sample_002', 'language', 'python'),
('tag_003', 'sample_003', 'priority', 'high');

-- ================================================================
-- PROMOTION SYSTEM TABLES (Phase 3 Extension)
-- ================================================================

-- Knowledge Promotion Queue for approval workflows
CREATE TABLE IF NOT EXISTS megamind_promotion_queue (
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
    original_content TEXT NOT NULL,
    target_chunk_id VARCHAR(50) DEFAULT NULL,
    promotion_session_id VARCHAR(50) NOT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (source_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    
    -- Indexes for workflow management
    INDEX idx_promotion_status (status, requested_at DESC),
    INDEX idx_promotion_reviewer (reviewed_by, reviewed_at DESC),
    INDEX idx_promotion_source (source_realm_id, source_chunk_id),
    INDEX idx_promotion_target (target_realm_id, status),
    INDEX idx_promotion_session (promotion_session_id, status)
) ENGINE=InnoDB;

-- Knowledge Promotion History for audit trail
CREATE TABLE IF NOT EXISTS megamind_promotion_history (
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
CREATE TABLE IF NOT EXISTS megamind_promotion_impact (
    impact_id VARCHAR(50) PRIMARY KEY,
    promotion_id VARCHAR(50) NOT NULL,
    
    -- Impact metrics
    affected_chunks_count INT DEFAULT 0,
    affected_relationships_count INT DEFAULT 0,
    potential_conflicts_count INT DEFAULT 0,
    
    -- Analysis results
    conflict_analysis JSON DEFAULT NULL,
    dependency_analysis JSON DEFAULT NULL,
    usage_impact JSON DEFAULT NULL,
    
    -- Quality assessment
    content_quality_score DECIMAL(3,2) DEFAULT NULL,
    relevance_score DECIMAL(3,2) DEFAULT NULL,
    uniqueness_score DECIMAL(3,2) DEFAULT NULL,
    
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