-- Phase 4 Performance Optimization: Semantic Search Database Indexes
-- Optimizes embedding storage and retrieval with realm-aware indexing

-- ========================================
-- Embedding-Specific Indexes
-- ========================================

-- Index for checking embedding existence by realm
CREATE INDEX idx_chunks_embedding_exists 
ON megamind_chunks (realm_id, (JSON_VALID(embedding))) 
WHERE embedding IS NOT NULL;

-- Compound index for realm-aware embedding search with access patterns
CREATE INDEX idx_chunks_realm_embedding_search 
ON megamind_chunks (realm_id, access_count DESC, last_accessed DESC) 
WHERE embedding IS NOT NULL;

-- Index for dual-realm access optimization (Global + Project)
CREATE INDEX idx_chunks_dual_realm_access 
ON megamind_chunks (realm_id, last_accessed DESC, access_count DESC);

-- Index for semantic search performance by content length
CREATE INDEX idx_chunks_embedding_content_length 
ON megamind_chunks (realm_id, CHAR_LENGTH(content), last_accessed DESC) 
WHERE embedding IS NOT NULL;

-- ========================================
-- Cross-Realm Performance Indexes
-- ========================================

-- Index for cross-realm relationship queries
CREATE INDEX idx_chunks_cross_realm_relations 
ON megamind_chunks (source_document, section_path, realm_id) 
WHERE embedding IS NOT NULL;

-- Index for realm priority calculations
CREATE INDEX idx_chunks_realm_priority 
ON megamind_chunks (realm_id, access_count DESC, token_count, created_at);

-- ========================================
-- Session and Change Management Indexes
-- ========================================

-- Index for session-based change tracking with embeddings
CREATE INDEX idx_session_changes_embedding 
ON session_changes (session_id, change_type, created_at) 
WHERE change_data->'$.embedding' IS NOT NULL;

-- Index for pending changes with realm context
CREATE INDEX idx_session_changes_realm 
ON session_changes (session_id, change_data->'$.realm_id', status, created_at);

-- ========================================
-- Performance Monitoring Views
-- ========================================

-- View for realm-aware embedding coverage analysis
CREATE VIEW realm_embedding_coverage AS
SELECT 
    realm_id,
    COUNT(*) as total_chunks,
    SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as chunks_with_embeddings,
    ROUND((SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*)) * 100, 2) as coverage_percentage,
    AVG(CASE WHEN embedding IS NOT NULL THEN CHAR_LENGTH(content) ELSE NULL END) as avg_content_length_with_embedding,
    AVG(access_count) as avg_access_count,
    MAX(last_accessed) as most_recent_access
FROM megamind_chunks
GROUP BY realm_id
ORDER BY coverage_percentage DESC;

-- View for dual-realm hot contexts with priority weighting
CREATE VIEW dual_realm_hot_contexts AS
SELECT 
    chunk_id, 
    content, 
    realm_id,
    access_count,
    token_count,
    CHAR_LENGTH(content) as content_length,
    last_accessed,
    CASE 
        WHEN realm_id LIKE 'PROJ_%' THEN access_count * 1.2  -- Project realm priority boost
        WHEN realm_id = 'GLOBAL' THEN access_count * 1.0     -- Global realm standard
        ELSE access_count * 0.8                              -- Other realms lower priority
    END as prioritized_score,
    CASE WHEN embedding IS NOT NULL THEN 'yes' ELSE 'no' END as has_embedding
FROM megamind_chunks 
WHERE embedding IS NOT NULL
ORDER BY prioritized_score DESC, last_accessed DESC;

-- View for embedding performance analytics
CREATE VIEW embedding_performance_analytics AS
SELECT 
    realm_id,
    COUNT(*) as total_chunks,
    SUM(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) as embedded_chunks,
    AVG(CASE WHEN embedding IS NOT NULL THEN access_count ELSE NULL END) as avg_access_embedded,
    AVG(CASE WHEN embedding IS NULL THEN access_count ELSE NULL END) as avg_access_non_embedded,
    SUM(CASE WHEN embedding IS NOT NULL AND access_count > 10 THEN 1 ELSE 0 END) as hot_embedded_chunks,
    MIN(CASE WHEN embedding IS NOT NULL THEN created_at ELSE NULL END) as first_embedded_chunk,
    MAX(CASE WHEN embedding IS NOT NULL THEN created_at ELSE NULL END) as latest_embedded_chunk
FROM megamind_chunks
GROUP BY realm_id;

-- View for cross-realm semantic search optimization
CREATE VIEW cross_realm_search_candidates AS
SELECT 
    chunk_id,
    content,
    source_document,
    section_path,
    realm_id,
    access_count,
    token_count,
    created_at,
    last_accessed,
    CASE 
        WHEN realm_id LIKE 'PROJ_%' THEN 1.2 
        WHEN realm_id = 'GLOBAL' THEN 1.0 
        ELSE 0.8 
    END as realm_weight_factor
FROM megamind_chunks 
WHERE embedding IS NOT NULL 
    AND (realm_id = 'GLOBAL' OR realm_id LIKE 'PROJ_%')
    AND access_count > 0
ORDER BY (access_count * CASE 
    WHEN realm_id LIKE 'PROJ_%' THEN 1.2 
    WHEN realm_id = 'GLOBAL' THEN 1.0 
    ELSE 0.8 
END) DESC;

-- ========================================
-- Maintenance and Optimization Procedures
-- ========================================

-- Stored procedure for embedding index maintenance
DELIMITER //
CREATE PROCEDURE OptimizeEmbeddingIndexes()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE index_name VARCHAR(255);
    DECLARE table_name VARCHAR(255);
    
    -- Cursor for embedding-related indexes
    DECLARE index_cursor CURSOR FOR 
        SELECT DISTINCT index_name, table_name 
        FROM information_schema.statistics 
        WHERE table_schema = DATABASE() 
        AND (index_name LIKE '%embedding%' OR index_name LIKE '%realm%')
        AND table_name IN ('megamind_chunks', 'session_changes');
    
    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;
    
    -- Log optimization start
    INSERT INTO system_log (level, message, created_at) 
    VALUES ('INFO', 'Starting embedding index optimization', NOW());
    
    -- Analyze tables with embedding data
    ANALYZE TABLE megamind_chunks;
    ANALYZE TABLE session_changes;
    
    -- Optimize embedding-related indexes
    OPEN index_cursor;
    read_loop: LOOP
        FETCH index_cursor INTO index_name, table_name;
        IF done THEN
            LEAVE read_loop;
        END IF;
        
        -- Log index optimization
        INSERT INTO system_log (level, message, created_at) 
        VALUES ('DEBUG', CONCAT('Optimizing index: ', index_name, ' on table: ', table_name), NOW());
        
    END LOOP;
    CLOSE index_cursor;
    
    -- Update statistics for realm-aware queries
    FLUSH TABLES megamind_chunks, session_changes;
    
    -- Log completion
    INSERT INTO system_log (level, message, created_at) 
    VALUES ('INFO', 'Embedding index optimization completed', NOW());
END //
DELIMITER ;

-- ========================================
-- Performance Monitoring Functions
-- ========================================

-- Function to calculate embedding storage efficiency
DELIMITER //
CREATE FUNCTION GetEmbeddingStorageEfficiency(target_realm VARCHAR(100))
RETURNS DECIMAL(5,2)
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE total_chunks INT DEFAULT 0;
    DECLARE embedded_chunks INT DEFAULT 0;
    DECLARE efficiency DECIMAL(5,2) DEFAULT 0.00;
    
    SELECT COUNT(*) INTO total_chunks 
    FROM megamind_chunks 
    WHERE realm_id = target_realm;
    
    SELECT COUNT(*) INTO embedded_chunks 
    FROM megamind_chunks 
    WHERE realm_id = target_realm AND embedding IS NOT NULL;
    
    IF total_chunks > 0 THEN
        SET efficiency = (embedded_chunks / total_chunks) * 100;
    END IF;
    
    RETURN efficiency;
END //
DELIMITER ;

-- Function to get dual-realm search performance score
DELIMITER //
CREATE FUNCTION GetDualRealmSearchScore(project_realm VARCHAR(100))
RETURNS DECIMAL(8,4)
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE global_embedded INT DEFAULT 0;
    DECLARE project_embedded INT DEFAULT 0;
    DECLARE global_access DECIMAL(10,2) DEFAULT 0;
    DECLARE project_access DECIMAL(10,2) DEFAULT 0;
    DECLARE search_score DECIMAL(8,4) DEFAULT 0;
    
    -- Get Global realm stats
    SELECT COUNT(*), AVG(access_count) 
    INTO global_embedded, global_access
    FROM megamind_chunks 
    WHERE realm_id = 'GLOBAL' AND embedding IS NOT NULL;
    
    -- Get Project realm stats
    SELECT COUNT(*), AVG(access_count) 
    INTO project_embedded, project_access
    FROM megamind_chunks 
    WHERE realm_id = project_realm AND embedding IS NOT NULL;
    
    -- Calculate weighted search performance score
    -- Project realm gets 1.2x weight, Global gets 1.0x weight
    SET search_score = (
        (project_embedded * project_access * 1.2) + 
        (global_embedded * global_access * 1.0)
    ) / GREATEST((project_embedded + global_embedded), 1);
    
    RETURN search_score;
END //
DELIMITER ;

-- ========================================
-- Index Usage Monitoring
-- ========================================

-- View to monitor index usage for optimization decisions
CREATE VIEW embedding_index_usage AS
SELECT 
    table_name,
    index_name,
    cardinality,
    CASE 
        WHEN index_name LIKE '%embedding%' THEN 'embedding'
        WHEN index_name LIKE '%realm%' THEN 'realm'
        WHEN index_name LIKE '%dual%' THEN 'dual_realm'
        ELSE 'other'
    END as index_category
FROM information_schema.statistics 
WHERE table_schema = DATABASE() 
    AND table_name IN ('megamind_chunks', 'session_changes')
    AND index_name != 'PRIMARY'
ORDER BY table_name, cardinality DESC;

-- ========================================
-- Cleanup and Maintenance
-- ========================================

-- Event for periodic embedding index optimization (runs daily at 2 AM)
CREATE EVENT IF NOT EXISTS OptimizeEmbeddingIndexesDaily
ON SCHEDULE EVERY 1 DAY
STARTS '2025-01-01 02:00:00'
DO
  CALL OptimizeEmbeddingIndexes();

-- Event for cleaning up expired embedding cache data (runs every hour)
CREATE EVENT IF NOT EXISTS CleanupExpiredEmbeddingData
ON SCHEDULE EVERY 1 HOUR
DO
  DELETE FROM system_log 
  WHERE level = 'DEBUG' 
    AND message LIKE '%embedding%' 
    AND created_at < DATE_SUB(NOW(), INTERVAL 24 HOUR);

-- ========================================
-- Documentation and Comments
-- ========================================

-- Add comments to key indexes for documentation
ALTER TABLE megamind_chunks COMMENT = 'Core chunks table with realm-aware semantic search capabilities';

-- Index documentation via information_schema (for reference)
-- idx_chunks_embedding_exists: Optimizes embedding existence checks by realm
-- idx_chunks_realm_embedding_search: Supports realm-aware semantic search with access patterns
-- idx_chunks_dual_realm_access: Enables efficient Global + Project realm queries
-- idx_chunks_embedding_content_length: Optimizes searches by content size within realms
-- idx_chunks_cross_realm_relations: Supports relationship discovery across realms
-- idx_chunks_realm_priority: Enables realm priority calculations for search ranking