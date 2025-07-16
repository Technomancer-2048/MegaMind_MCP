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
    source_realm_id VARCHAR(50) NOT NULL,
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

-- ================================================================
-- ENHANCED SESSION SYSTEM (Issue #15 - Phase 1)
-- ================================================================

-- Enhanced Session Management with State Tracking
CREATE TABLE IF NOT EXISTS megamind_sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    
    -- Session identification and context
    session_name VARCHAR(255) DEFAULT NULL,
    user_id VARCHAR(100) DEFAULT NULL,
    realm_id VARCHAR(50) NOT NULL DEFAULT 'PROJECT',
    project_context VARCHAR(255) DEFAULT NULL,
    
    -- Session state management (open/active/archived)
    session_state ENUM('open', 'active', 'archived') NOT NULL DEFAULT 'open',
    state_changed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    -- Session metadata and configuration
    session_config JSON DEFAULT NULL,
    session_tags JSON DEFAULT NULL,
    priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    
    -- Semantic indexing support
    enable_semantic_indexing BOOLEAN DEFAULT TRUE,
    content_token_limit INT DEFAULT 128,
    embedding_generation_enabled BOOLEAN DEFAULT TRUE,
    
    -- Session statistics
    total_entries INT DEFAULT 0,
    total_chunks_accessed INT DEFAULT 0,
    total_operations INT DEFAULT 0,
    last_semantic_update TIMESTAMP DEFAULT NULL,
    
    -- Lifecycle timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    archived_at TIMESTAMP DEFAULT NULL,
    
    -- Performance and analytics
    performance_score DECIMAL(3,2) DEFAULT 0.0,
    context_quality_score DECIMAL(3,2) DEFAULT 0.0,
    
    -- Indexes for efficient session management
    INDEX idx_session_state (session_state, last_activity DESC),
    INDEX idx_session_realm (realm_id, session_state),
    INDEX idx_session_user (user_id, created_at DESC),
    INDEX idx_session_activity (last_activity DESC),
    INDEX idx_session_semantic (enable_semantic_indexing, last_semantic_update),
    INDEX idx_session_priority (priority, session_state)
) ENGINE=InnoDB;

-- Session Activity Entries with Semantic Context
CREATE TABLE IF NOT EXISTS megamind_session_entries (
    entry_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    
    -- Entry classification and metadata
    entry_type ENUM('query', 'operation', 'result', 'context_switch', 'error', 'system_event') NOT NULL,
    operation_type VARCHAR(100) DEFAULT NULL,
    entry_content TEXT NOT NULL,
    
    -- Token management for sentence-transformers
    content_tokens INT DEFAULT 0,
    content_truncated BOOLEAN DEFAULT FALSE,
    original_content_hash VARCHAR(64) DEFAULT NULL,
    
    -- Semantic indexing and relationships
    semantic_summary TEXT DEFAULT NULL,
    related_chunk_ids JSON DEFAULT NULL,
    context_relevance_score DECIMAL(3,2) DEFAULT 0.0,
    
    -- Entry hierarchy and sequence
    parent_entry_id VARCHAR(50) DEFAULT NULL,
    entry_sequence INT NOT NULL,
    conversation_turn INT DEFAULT 1,
    
    -- Performance and quality metrics
    processing_time_ms INT DEFAULT 0,
    success_indicator BOOLEAN DEFAULT TRUE,
    quality_score DECIMAL(3,2) DEFAULT 0.0,
    
    -- Temporal data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata and additional context
    entry_metadata JSON DEFAULT NULL,
    user_feedback JSON DEFAULT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (session_id) REFERENCES megamind_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_entry_id) REFERENCES megamind_session_entries(entry_id) ON DELETE SET NULL,
    
    -- Indexes for efficient entry management
    INDEX idx_entries_session (session_id, entry_sequence),
    INDEX idx_entries_type (entry_type, created_at DESC),
    INDEX idx_entries_sequence (session_id, conversation_turn, entry_sequence),
    INDEX idx_entries_semantic (context_relevance_score DESC, created_at DESC),
    INDEX idx_entries_parent (parent_entry_id, entry_sequence),
    INDEX idx_entries_tokens (content_tokens, content_truncated),
    INDEX idx_entries_performance (processing_time_ms DESC, success_indicator)
) ENGINE=InnoDB;

-- Session Embeddings for Semantic Search and Context Preservation
CREATE TABLE IF NOT EXISTS megamind_session_embeddings (
    embedding_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    entry_id VARCHAR(50) DEFAULT NULL,
    
    -- Embedding metadata
    embedding_type ENUM('session_summary', 'entry_content', 'context_window', 'conversation_turn') NOT NULL,
    content_source TEXT NOT NULL,
    content_tokens INT NOT NULL,
    
    -- Token-aware content strategy (128 optimal, 256 hard limit)
    token_limit_applied INT DEFAULT 128,
    content_truncated BOOLEAN DEFAULT FALSE,
    truncation_strategy ENUM('head', 'tail', 'middle', 'smart_summary') DEFAULT 'smart_summary',
    
    -- Vector embedding storage
    embedding_vector JSON NOT NULL,
    model_name VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_dimension INT DEFAULT 384,
    
    -- Quality and relevance metrics
    embedding_quality_score DECIMAL(3,2) DEFAULT 0.0,
    semantic_density DECIMAL(3,2) DEFAULT 0.0,
    context_preservation_score DECIMAL(3,2) DEFAULT 0.0,
    
    -- Temporal and lifecycle data
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INT DEFAULT 0,
    
    -- Multi-level embedding strategy support
    aggregation_level ENUM('entry', 'turn', 'session', 'cross_session') DEFAULT 'entry',
    parent_embedding_id VARCHAR(50) DEFAULT NULL,
    
    -- Foreign key constraints
    FOREIGN KEY (session_id) REFERENCES megamind_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (entry_id) REFERENCES megamind_session_entries(entry_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_embedding_id) REFERENCES megamind_session_embeddings(embedding_id) ON DELETE SET NULL,
    
    -- Indexes for semantic search and retrieval
    INDEX idx_session_embeddings_session (session_id, embedding_type),
    INDEX idx_session_embeddings_entry (entry_id, embedding_type),
    INDEX idx_session_embeddings_quality (embedding_quality_score DESC, semantic_density DESC),
    INDEX idx_session_embeddings_access (access_count DESC, last_accessed DESC),
    INDEX idx_session_embeddings_tokens (content_tokens, token_limit_applied),
    INDEX idx_session_embeddings_hierarchy (parent_embedding_id, aggregation_level),
    INDEX idx_session_embeddings_model (model_name, embedding_dimension)
) ENGINE=InnoDB;

-- Session Context Windows for Dynamic Context Assembly
CREATE TABLE IF NOT EXISTS megamind_session_context_windows (
    window_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    
    -- Context window metadata
    window_type ENUM('conversation', 'operation_sequence', 'semantic_cluster', 'temporal_window') NOT NULL,
    window_size INT NOT NULL,
    start_entry_id VARCHAR(50) NOT NULL,
    end_entry_id VARCHAR(50) NOT NULL,
    
    -- Content and token management
    aggregated_content TEXT NOT NULL,
    total_tokens INT NOT NULL,
    content_summary TEXT DEFAULT NULL,
    
    -- Semantic coherence metrics
    coherence_score DECIMAL(3,2) DEFAULT 0.0,
    relevance_score DECIMAL(3,2) DEFAULT 0.0,
    context_quality DECIMAL(3,2) DEFAULT 0.0,
    
    -- Window relationships and hierarchy
    parent_window_id VARCHAR(50) DEFAULT NULL,
    child_windows_count INT DEFAULT 0,
    
    -- Lifecycle and usage tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    usage_count INT DEFAULT 0,
    
    -- Foreign key constraints
    FOREIGN KEY (session_id) REFERENCES megamind_sessions(session_id) ON DELETE CASCADE,
    FOREIGN KEY (start_entry_id) REFERENCES megamind_session_entries(entry_id) ON DELETE CASCADE,
    FOREIGN KEY (end_entry_id) REFERENCES megamind_session_entries(entry_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_window_id) REFERENCES megamind_session_context_windows(window_id) ON DELETE SET NULL,
    
    -- Indexes for context window management
    INDEX idx_context_windows_session (session_id, window_type),
    INDEX idx_context_windows_range (start_entry_id, end_entry_id),
    INDEX idx_context_windows_quality (coherence_score DESC, relevance_score DESC),
    INDEX idx_context_windows_usage (usage_count DESC, last_used DESC),
    INDEX idx_context_windows_tokens (total_tokens, window_size),
    INDEX idx_context_windows_hierarchy (parent_window_id, child_windows_count)
) ENGINE=InnoDB;

-- ================================================================
-- SAMPLE DATA FOR ENHANCED SESSION SYSTEM TESTING
-- ================================================================

-- Insert sample session for testing the new session system
INSERT IGNORE INTO megamind_sessions (
    session_id, session_name, user_id, realm_id, project_context, 
    session_state, session_config, priority, enable_semantic_indexing, 
    content_token_limit, total_entries, created_at
) VALUES (
    'test_session_001', 
    'Sample Development Session', 
    'test_user', 
    'MegaMind_MCP', 
    'Enhanced Session System Development',
    'active',
    '{"session_type": "development", "features": ["semantic_indexing", "token_management"]}',
    'high',
    TRUE,
    128,
    0,
    CURRENT_TIMESTAMP
);

-- Insert sample session entry
INSERT IGNORE INTO megamind_session_entries (
    entry_id, session_id, entry_type, operation_type, entry_content,
    content_tokens, semantic_summary, entry_sequence, conversation_turn,
    processing_time_ms, success_indicator, quality_score, created_at
) VALUES (
    'entry_001',
    'test_session_001',
    'query',
    'search_chunks',
    'Search for database schema information related to session management',
    12,
    'User searching for session management database schema details',
    1,
    1,
    250,
    TRUE,
    0.85,
    CURRENT_TIMESTAMP
);

-- ================================================================
-- PHASE 2: ENHANCED MULTI-EMBEDDING ENTRY SYSTEM
-- Session Tracking System Integration
-- ================================================================

-- Enhanced Multi-Embedding Entry System - Phase 1 Foundation Tables
-- These tables support the advanced content processing capabilities

-- Document structure analysis table
CREATE TABLE IF NOT EXISTS megamind_document_structures (
    document_id VARCHAR(64) PRIMARY KEY,
    realm_id VARCHAR(64) NOT NULL DEFAULT 'PROJECT',
    document_name VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    content_type ENUM('markdown', 'text', 'code', 'mixed') NOT NULL DEFAULT 'text',
    structure_analysis JSON NOT NULL,
    element_count INT NOT NULL DEFAULT 0,
    semantic_boundary_count INT NOT NULL DEFAULT 0,
    processing_statistics JSON,
    table_preservation_applied BOOLEAN DEFAULT FALSE,
    sentence_boundary_detection BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_document_realm (realm_id),
    INDEX idx_document_type (content_type),
    INDEX idx_document_hash (content_hash),
    INDEX idx_document_updated (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Chunk metadata with enhanced processing information
CREATE TABLE IF NOT EXISTS megamind_chunk_metadata (
    metadata_id VARCHAR(64) PRIMARY KEY,
    chunk_id VARCHAR(64) NOT NULL,
    document_id VARCHAR(64) NOT NULL,
    chunk_type ENUM('paragraph', 'heading', 'list', 'code_block', 'table', 'quote', 'mixed') NOT NULL DEFAULT 'paragraph',
    processing_strategy ENUM('semantic_aware', 'markdown_structure', 'hybrid', 'table_preservation') NOT NULL DEFAULT 'semantic_aware',
    token_count INT NOT NULL DEFAULT 0,
    line_start INT NOT NULL DEFAULT 1,
    line_end INT NOT NULL DEFAULT 1,
    quality_score DECIMAL(3,2) DEFAULT 0.0,
    complexity_indicators JSON,
    boundary_analysis JSON,
    chunk_statistics JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_chunk_meta_chunk (chunk_id),
    INDEX idx_chunk_meta_document (document_id),
    INDEX idx_chunk_meta_type (chunk_type),
    INDEX idx_chunk_meta_strategy (processing_strategy),
    INDEX idx_chunk_meta_quality (quality_score),
    
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (document_id) REFERENCES megamind_document_structures(document_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Entry embeddings with model and optimization information
CREATE TABLE IF NOT EXISTS megamind_entry_embeddings (
    embedding_id VARCHAR(64) PRIMARY KEY,
    chunk_id VARCHAR(64) NOT NULL,
    realm_id VARCHAR(64) NOT NULL DEFAULT 'PROJECT',
    embedding_vector JSON NOT NULL,
    model_name VARCHAR(100) DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    model_version VARCHAR(50) DEFAULT '1.0',
    embedding_dimension INT DEFAULT 384,
    optimization_applied BOOLEAN DEFAULT FALSE,
    optimization_level ENUM('minimal', 'standard', 'aggressive') DEFAULT 'standard',
    text_preprocessing JSON,
    compression_ratio DECIMAL(4,2) DEFAULT 1.0,
    quality_metrics JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    access_count INT DEFAULT 0,
    
    INDEX idx_entry_embedding_chunk (chunk_id),
    INDEX idx_entry_embedding_realm (realm_id),
    INDEX idx_entry_embedding_model (model_name, model_version),
    INDEX idx_entry_embedding_dimension (embedding_dimension),
    INDEX idx_entry_embedding_quality (optimization_applied, compression_ratio),
    INDEX idx_entry_embedding_access (access_count DESC, last_accessed DESC),
    
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Quality assessments with 8-dimensional scoring
CREATE TABLE IF NOT EXISTS megamind_quality_assessments (
    assessment_id VARCHAR(64) PRIMARY KEY,
    chunk_id VARCHAR(64) NOT NULL,
    assessment_type ENUM('automatic', 'manual', 'hybrid') DEFAULT 'automatic',
    overall_score DECIMAL(3,2) NOT NULL,
    quality_level ENUM('excellent', 'good', 'acceptable', 'poor') NOT NULL,
    
    -- 8-dimensional quality scores
    readability_score DECIMAL(3,2) DEFAULT 0.0,
    technical_accuracy_score DECIMAL(3,2) DEFAULT 0.0,
    completeness_score DECIMAL(3,2) DEFAULT 0.0,
    relevance_score DECIMAL(3,2) DEFAULT 0.0,
    freshness_score DECIMAL(3,2) DEFAULT 0.0,
    coherence_score DECIMAL(3,2) DEFAULT 0.0,
    uniqueness_score DECIMAL(3,2) DEFAULT 0.0,
    authority_score DECIMAL(3,2) DEFAULT 0.0,
    
    -- Assessment metadata
    assessment_confidence DECIMAL(3,2) DEFAULT 0.0,
    issues_identified JSON,
    improvement_suggestions JSON,
    assessment_context JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    assessor_id VARCHAR(100) DEFAULT 'ai_quality_assessor',
    
    INDEX idx_quality_chunk (chunk_id),
    INDEX idx_quality_overall (overall_score DESC),
    INDEX idx_quality_level (quality_level),
    INDEX idx_quality_confidence (assessment_confidence DESC),
    INDEX idx_quality_created (created_at DESC),
    
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- System configuration for Enhanced Entry System
CREATE TABLE IF NOT EXISTS megamind_system_config (
    config_id VARCHAR(64) PRIMARY KEY,
    config_key VARCHAR(255) UNIQUE NOT NULL,
    config_value TEXT NOT NULL,
    config_type ENUM('string', 'number', 'boolean', 'json') DEFAULT 'string',
    description TEXT,
    is_system_config BOOLEAN DEFAULT FALSE,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_config_key (config_key),
    INDEX idx_config_type (config_type),
    INDEX idx_config_system (is_system_config)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ================================================================
-- PHASE 2: SESSION TRACKING SYSTEM TABLES
-- ================================================================

-- Session tracking table for Enhanced Multi-Embedding Entry System
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
    INDEX idx_session_activity (last_activity)
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
-- ENHANCED EXISTING TABLES FOR SESSION AWARENESS
-- ================================================================

-- Add session tracking to document structures
ALTER TABLE megamind_document_structures
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS session_metadata JSON,
ADD INDEX IF NOT EXISTS idx_document_session (session_id);

-- Add session tracking to chunk metadata  
ALTER TABLE megamind_chunk_metadata
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS session_operation VARCHAR(50),
ADD INDEX IF NOT EXISTS idx_chunk_meta_session (session_id);

-- Add session tracking to embeddings
ALTER TABLE megamind_entry_embeddings
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD COLUMN IF NOT EXISTS session_metadata JSON,
ADD INDEX IF NOT EXISTS idx_embedding_session (session_id);

-- Add session tracking to quality assessments
ALTER TABLE megamind_quality_assessments
ADD COLUMN IF NOT EXISTS session_id VARCHAR(64),
ADD INDEX IF NOT EXISTS idx_quality_session (session_id);

-- ================================================================
-- VIEWS FOR SESSION OPERATIONS
-- ================================================================

-- Active sessions view
CREATE OR REPLACE VIEW megamind_active_sessions_view AS
SELECT 
    s.session_id,
    s.session_type,
    s.realm_id,
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
-- PHASE 2 SYSTEM CONFIGURATION
-- ================================================================

-- Insert enhanced entry system configuration
INSERT IGNORE INTO megamind_system_config (config_id, config_key, config_value, config_type, description, is_system_config) VALUES
-- Content Processing Configuration
('cfg_001', 'content.max_tokens_per_chunk', '512', 'number', 'Maximum tokens per chunk for content processing', TRUE),
('cfg_002', 'content.min_tokens_per_chunk', '50', 'number', 'Minimum tokens per chunk for content processing', TRUE),
('cfg_003', 'content.overlap_tokens', '50', 'number', 'Token overlap between chunks', TRUE),
('cfg_004', 'content.quality_threshold', '0.7', 'number', 'Minimum quality threshold for chunk acceptance', TRUE),
('cfg_005', 'content.enable_table_preservation', 'true', 'boolean', 'Enable table structure preservation in markdown', TRUE),
('cfg_006', 'content.enable_sentence_boundary_detection', 'true', 'boolean', 'Enable intelligent sentence boundary detection', TRUE),

-- Embedding Configuration
('cfg_007', 'embedding.default_model', 'sentence-transformers/all-MiniLM-L6-v2', 'string', 'Default embedding model', TRUE),
('cfg_008', 'embedding.default_dimension', '384', 'number', 'Default embedding dimension', TRUE),
('cfg_009', 'embedding.batch_size', '32', 'number', 'Default batch size for embedding generation', TRUE),
('cfg_010', 'embedding.optimization_level', 'standard', 'string', 'Default text optimization level (minimal, standard, aggressive)', TRUE),

-- Quality Assessment Configuration
('cfg_011', 'quality.enable_8d_scoring', 'true', 'boolean', 'Enable 8-dimensional quality scoring', TRUE),
('cfg_012', 'quality.readability_weight', '0.15', 'number', 'Weight for readability dimension', TRUE),
('cfg_013', 'quality.technical_accuracy_weight', '0.25', 'number', 'Weight for technical accuracy dimension', TRUE),
('cfg_014', 'quality.completeness_weight', '0.20', 'number', 'Weight for completeness dimension', TRUE),
('cfg_015', 'quality.relevance_weight', '0.15', 'number', 'Weight for relevance dimension', TRUE),
('cfg_016', 'quality.freshness_weight', '0.10', 'number', 'Weight for freshness dimension', TRUE),
('cfg_017', 'quality.coherence_weight', '0.10', 'number', 'Weight for coherence dimension', TRUE),
('cfg_018', 'quality.uniqueness_weight', '0.03', 'number', 'Weight for uniqueness dimension', TRUE),
('cfg_019', 'quality.authority_weight', '0.02', 'number', 'Weight for authority dimension', TRUE),

-- Session Management Configuration  
('cfg_020', 'session.default_timeout_minutes', '120', 'number', 'Default session timeout in minutes', TRUE),
('cfg_021', 'session.max_concurrent_sessions', '10', 'number', 'Maximum concurrent active sessions per realm', TRUE),
('cfg_022', 'session.chunk_batch_size', '100', 'number', 'Default batch size for chunk processing', TRUE),
('cfg_023', 'session.embedding_batch_size', '32', 'number', 'Default batch size for embedding generation', TRUE),
('cfg_024', 'session.auto_save_interval_seconds', '300', 'number', 'Auto-save session state interval', TRUE);

-- ================================================================
-- SAMPLE DATA FOR PHASE 2 TESTING
-- ================================================================

-- Insert sample enhanced entry system session
INSERT IGNORE INTO megamind_embedding_sessions (
    session_id, session_type, realm_id, created_by, metadata,
    total_chunks_processed, total_embeddings_generated
) VALUES (
    'phase2_test_session',
    'analysis',
    'MegaMind_MCP',
    'phase2_testing_system',
    '{"purpose": "Phase 2 Enhanced Entry System Testing", "features": ["content_analysis", "intelligent_chunking", "quality_assessment", "embedding_optimization"]}',
    0,
    0
);

-- Schema update completion message
SELECT 'Phase 2 Enhanced Multi-Embedding Entry System schema integrated successfully into container init schema' as status;