-- Phase 3: Knowledge Management and Session Tracking Schema
-- Enhanced Multi-Embedding Entry System
-- Tables for knowledge management, operational session tracking, and retrieval optimization

-- 1. Knowledge Documents Table
CREATE TABLE IF NOT EXISTS megamind_knowledge_documents (
    document_id VARCHAR(50) PRIMARY KEY,
    source_path TEXT NOT NULL,
    title VARCHAR(255) NOT NULL,
    knowledge_type ENUM('documentation', 'code_pattern', 'best_practice', 'troubleshooting', 'architectural', 'configuration', 'general') NOT NULL,
    metadata JSON,
    tags JSON,
    ingested_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    version INT DEFAULT 1,
    realm_id VARCHAR(50),
    INDEX idx_knowledge_type (knowledge_type),
    INDEX idx_ingested_date (ingested_date),
    INDEX idx_realm (realm_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 2. Knowledge Chunks Table (extends existing chunks)
CREATE TABLE IF NOT EXISTS megamind_knowledge_chunks (
    chunk_id VARCHAR(50) PRIMARY KEY,
    document_id VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    chunk_type VARCHAR(50) NOT NULL,
    knowledge_type ENUM('documentation', 'code_pattern', 'best_practice', 'troubleshooting', 'architectural', 'configuration', 'general') NOT NULL,
    quality_score FLOAT DEFAULT 0.0,
    importance_score FLOAT DEFAULT 0.5,
    confidence_score FLOAT DEFAULT 0.8,
    tags JSON,
    metadata JSON,
    usage_count INT DEFAULT 0,
    last_accessed TIMESTAMP NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES megamind_knowledge_documents(document_id) ON DELETE CASCADE,
    INDEX idx_document (document_id),
    INDEX idx_quality (quality_score),
    INDEX idx_importance (importance_score),
    INDEX idx_usage (usage_count),
    INDEX idx_last_accessed (last_accessed)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 3. Knowledge Relationships Table
CREATE TABLE IF NOT EXISTS megamind_knowledge_relationships (
    relationship_id VARCHAR(50) PRIMARY KEY,
    source_chunk_id VARCHAR(50) NOT NULL,
    target_chunk_id VARCHAR(50) NOT NULL,
    relationship_type ENUM('parent_child', 'sibling', 'cross_reference', 'prerequisite', 'alternative', 'example_of', 'implements', 'depends_on') NOT NULL,
    confidence FLOAT DEFAULT 1.0,
    discovered_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON,
    FOREIGN KEY (source_chunk_id) REFERENCES megamind_knowledge_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (target_chunk_id) REFERENCES megamind_knowledge_chunks(chunk_id) ON DELETE CASCADE,
    UNIQUE KEY unique_relationship (source_chunk_id, target_chunk_id, relationship_type),
    INDEX idx_source (source_chunk_id),
    INDEX idx_target (target_chunk_id),
    INDEX idx_type (relationship_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 4. Knowledge Clusters Table
CREATE TABLE IF NOT EXISTS megamind_knowledge_clusters (
    cluster_id VARCHAR(50) PRIMARY KEY,
    cluster_name VARCHAR(255),
    cluster_type VARCHAR(50),
    chunk_ids JSON NOT NULL,
    centroid_chunk_id VARCHAR(50),
    metadata JSON,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_created (created_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 5. Operational Sessions Table
CREATE TABLE IF NOT EXISTS megamind_operational_sessions (
    session_id VARCHAR(50) PRIMARY KEY,
    session_type VARCHAR(50) NOT NULL DEFAULT 'general',
    user_id VARCHAR(100),
    description TEXT,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    total_actions INT DEFAULT 0,
    accomplishments JSON,
    pending_tasks JSON,
    metadata JSON,
    INDEX idx_user (user_id),
    INDEX idx_start_time (start_time),
    INDEX idx_session_type (session_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 6. Session Actions Table
CREATE TABLE IF NOT EXISTS megamind_session_actions (
    action_id VARCHAR(100) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    action_type ENUM('search', 'create_chunk', 'update_chunk', 'delete_chunk', 'analyze_document', 'generate_embedding', 'assess_quality', 'discover_relationship', 'retrieve_knowledge', 'promote_chunk', 'custom') NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT NOT NULL,
    details JSON,
    result TEXT,
    priority ENUM('critical', 'high', 'medium', 'low') DEFAULT 'medium',
    error TEXT,
    duration_ms INT,
    FOREIGN KEY (session_id) REFERENCES megamind_operational_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session (session_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_action_type (action_type),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 7. Session Context Table
CREATE TABLE IF NOT EXISTS megamind_session_context (
    context_id VARCHAR(50) PRIMARY KEY,
    session_id VARCHAR(50) NOT NULL,
    context_type VARCHAR(50) NOT NULL,
    context_data JSON NOT NULL,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES megamind_operational_sessions(session_id) ON DELETE CASCADE,
    INDEX idx_session (session_id),
    INDEX idx_type (context_type)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 8. Retrieval Optimization Table
CREATE TABLE IF NOT EXISTS megamind_retrieval_optimization (
    optimization_id VARCHAR(50) PRIMARY KEY,
    optimization_type ENUM('hot_chunks', 'cache_recommendation', 'prefetch_pattern', 'access_sequence') NOT NULL,
    chunk_ids JSON,
    metadata JSON,
    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_date TIMESTAMP NULL,
    INDEX idx_type (optimization_type),
    INDEX idx_created (created_date),
    INDEX idx_expires (expires_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 9. Knowledge Usage Analytics Table
CREATE TABLE IF NOT EXISTS megamind_knowledge_usage (
    usage_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    session_id VARCHAR(50),
    access_type ENUM('search', 'retrieve', 'reference', 'update') NOT NULL,
    access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id VARCHAR(100),
    context JSON,
    FOREIGN KEY (chunk_id) REFERENCES megamind_knowledge_chunks(chunk_id) ON DELETE CASCADE,
    INDEX idx_chunk (chunk_id),
    INDEX idx_session (session_id),
    INDEX idx_access_time (access_time),
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- 10. Create views for easier querying

-- View: Active sessions with summary
CREATE OR REPLACE VIEW megamind_active_sessions_view AS
SELECT 
    s.session_id,
    s.session_type,
    s.user_id,
    s.description,
    s.start_time,
    s.total_actions,
    COUNT(DISTINCT a.action_id) as actual_action_count,
    MAX(a.timestamp) as last_action_time,
    JSON_LENGTH(s.accomplishments) as accomplishment_count,
    JSON_LENGTH(s.pending_tasks) as pending_task_count
FROM megamind_operational_sessions s
LEFT JOIN megamind_session_actions a ON s.session_id = a.session_id
WHERE s.end_time IS NULL
GROUP BY s.session_id;

-- View: Knowledge chunk relationships with details
CREATE OR REPLACE VIEW megamind_knowledge_graph_view AS
SELECT 
    r.relationship_id,
    r.source_chunk_id,
    s.content as source_content_preview,
    s.knowledge_type as source_type,
    r.target_chunk_id,
    t.content as target_content_preview,
    t.knowledge_type as target_type,
    r.relationship_type,
    r.confidence,
    r.discovered_date
FROM megamind_knowledge_relationships r
JOIN megamind_knowledge_chunks s ON r.source_chunk_id = s.chunk_id
JOIN megamind_knowledge_chunks t ON r.target_chunk_id = t.chunk_id;

-- View: Hot chunks based on usage
CREATE OR REPLACE VIEW megamind_hot_chunks_view AS
SELECT 
    c.chunk_id,
    c.document_id,
    c.knowledge_type,
    c.quality_score,
    c.importance_score,
    c.usage_count,
    c.last_accessed,
    COUNT(u.usage_id) as recent_access_count,
    SUBSTRING(c.content, 1, 200) as content_preview
FROM megamind_knowledge_chunks c
LEFT JOIN megamind_knowledge_usage u ON c.chunk_id = u.chunk_id 
    AND u.access_time > DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY c.chunk_id
ORDER BY recent_access_count DESC, c.usage_count DESC
LIMIT 100;

-- 11. Stored procedures for common operations

DELIMITER //

-- Procedure: Update chunk usage statistics
CREATE PROCEDURE update_chunk_usage_stats(
    IN p_chunk_id VARCHAR(50),
    IN p_session_id VARCHAR(50),
    IN p_access_type VARCHAR(50)
)
BEGIN
    -- Update usage count and last accessed
    UPDATE megamind_knowledge_chunks 
    SET usage_count = usage_count + 1,
        last_accessed = CURRENT_TIMESTAMP
    WHERE chunk_id = p_chunk_id;
    
    -- Insert usage record
    INSERT INTO megamind_knowledge_usage (
        usage_id, chunk_id, session_id, access_type
    ) VALUES (
        CONCAT('usage_', MD5(CONCAT(p_chunk_id, NOW(), RAND()))),
        p_chunk_id, p_session_id, p_access_type
    );
END//

-- Procedure: Get session summary with actions
CREATE PROCEDURE get_session_summary(
    IN p_session_id VARCHAR(50)
)
BEGIN
    -- Get session details
    SELECT * FROM megamind_operational_sessions WHERE session_id = p_session_id;
    
    -- Get action summary
    SELECT 
        action_type,
        COUNT(*) as count,
        AVG(duration_ms) as avg_duration_ms,
        MAX(timestamp) as last_occurrence
    FROM megamind_session_actions
    WHERE session_id = p_session_id
    GROUP BY action_type;
    
    -- Get recent actions
    SELECT * FROM megamind_session_actions
    WHERE session_id = p_session_id
    ORDER BY timestamp DESC
    LIMIT 10;
END//

DELIMITER ;

-- 12. Indexes for performance optimization
CREATE INDEX idx_knowledge_search ON megamind_knowledge_chunks(knowledge_type, quality_score DESC, importance_score DESC);
CREATE INDEX idx_session_search ON megamind_session_actions(session_id, timestamp DESC, priority);
CREATE INDEX idx_usage_analytics ON megamind_knowledge_usage(chunk_id, access_time DESC);

-- 13. Initial configuration entries
INSERT INTO megamind_system_config (config_key, config_value, description) VALUES
    ('phase3.knowledge.default_similarity_threshold', '0.7', 'Default similarity threshold for relationship discovery'),
    ('phase3.knowledge.max_chunk_relationships', '50', 'Maximum relationships per chunk'),
    ('phase3.session.max_history_size', '1000', 'Maximum session history size'),
    ('phase3.session.auto_close_hours', '24', 'Hours before auto-closing inactive sessions'),
    ('phase3.optimization.hot_chunk_threshold', '10', 'Minimum accesses to be considered hot'),
    ('phase3.optimization.cache_size', '100', 'Maximum chunks in optimization cache')
ON DUPLICATE KEY UPDATE 
    config_value = VALUES(config_value),
    updated_at = CURRENT_TIMESTAMP;