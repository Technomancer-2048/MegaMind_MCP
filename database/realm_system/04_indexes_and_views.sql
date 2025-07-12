-- Performance Indexes and Inheritance Views
-- Phase 1: Realm-Aware Query Optimization
-- Database: megamind_database (MySQL 8.0+)

-- Additional performance indexes for realm operations
CREATE INDEX idx_chunks_realm_embedding_search ON megamind_chunks (realm_id, access_count DESC) WHERE embedding IS NOT NULL;
CREATE INDEX idx_chunks_dual_realm_access ON megamind_chunks (realm_id, last_accessed DESC);
CREATE INDEX idx_chunks_realm_content_search ON megamind_chunks (realm_id, source_document, section_path);

-- Cross-realm relationship performance indexes
CREATE INDEX idx_relationships_cross_realm_discovery ON megamind_chunk_relationships (is_cross_realm, strength DESC, created_at DESC);
CREATE INDEX idx_relationships_realm_pair ON megamind_chunk_relationships (source_realm_id, target_realm_id, relationship_type);

-- Session management performance indexes
CREATE INDEX idx_session_changes_impact ON megamind_session_changes (impact_score DESC, timestamp);
CREATE INDEX idx_session_changes_type ON megamind_session_changes (change_type, status);
CREATE INDEX idx_session_changes_realm_priority ON megamind_session_changes (source_realm_id, impact_score DESC);

-- Virtual view for realm-aware chunk access with inheritance
CREATE VIEW megamind_chunks_with_inheritance AS
SELECT 
    c.chunk_id,
    c.content,
    c.source_document,
    c.section_path,
    c.chunk_type,
    c.realm_id AS source_realm_id,
    c.access_count,
    c.last_accessed,
    c.created_at,
    c.embedding,
    r.realm_name AS source_realm_name,
    r.realm_type AS source_realm_type,
    CASE 
        WHEN c.realm_id = @current_realm THEN 'direct'
        WHEN c.realm_id = 'GLOBAL' THEN 'inherited_global'
        ELSE 'inherited_project'
    END AS access_type,
    CASE 
        WHEN c.realm_id = @current_realm THEN 1.0
        WHEN c.realm_id = 'GLOBAL' THEN 0.8
        ELSE 0.6
    END AS inheritance_weight
FROM megamind_chunks c
JOIN megamind_realms r ON c.realm_id = r.realm_id
WHERE c.realm_id = @current_realm
   OR c.realm_id = 'GLOBAL'
   OR c.realm_id IN (
       SELECT ri.parent_realm_id 
       FROM megamind_realm_inheritance ri 
       WHERE ri.child_realm_id = @current_realm 
         AND ri.inheritance_type IN ('full', 'selective')
         AND ri.is_active = TRUE
   );

-- View for realm coverage and statistics
CREATE VIEW realm_coverage_stats AS
SELECT 
    r.realm_id,
    r.realm_name,
    r.realm_type,
    COUNT(c.chunk_id) as total_chunks,
    SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) as chunks_with_embeddings,
    AVG(c.access_count) as avg_access_count,
    MAX(c.last_accessed) as last_chunk_access,
    COUNT(DISTINCT c.source_document) as unique_documents,
    (SUM(CASE WHEN c.embedding IS NOT NULL THEN 1 ELSE 0 END) / COUNT(c.chunk_id)) * 100 as embedding_coverage_percentage
FROM megamind_realms r
LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
WHERE r.is_active = TRUE
GROUP BY r.realm_id, r.realm_name, r.realm_type;

-- View for cross-realm relationship analysis
CREATE VIEW cross_realm_relationships AS
SELECT 
    r1.realm_name as source_realm,
    r2.realm_name as target_realm,
    cr.relationship_type,
    COUNT(*) as relationship_count,
    AVG(cr.strength) as avg_strength,
    MAX(cr.created_at) as latest_relationship
FROM megamind_chunk_relationships cr
JOIN megamind_chunks c1 ON cr.chunk_id = c1.chunk_id
JOIN megamind_chunks c2 ON cr.related_chunk_id = c2.chunk_id
JOIN megamind_realms r1 ON cr.source_realm_id = r1.realm_id
JOIN megamind_realms r2 ON cr.target_realm_id = r2.realm_id
WHERE cr.is_cross_realm = TRUE
GROUP BY r1.realm_name, r2.realm_name, cr.relationship_type;

-- View for dual-realm hot contexts (optimized for environment-based access)
CREATE VIEW dual_realm_hot_contexts AS
SELECT 
    c.chunk_id, 
    c.content, 
    c.realm_id,
    c.access_count,
    c.last_accessed,
    r.realm_name,
    CASE 
        WHEN c.realm_id = @current_realm THEN c.access_count * 1.2  -- Project realm priority boost
        WHEN c.realm_id = 'GLOBAL' THEN c.access_count * 1.0        -- Global realm standard weight
        ELSE c.access_count * 0.8                                  -- Other inherited realms
    END as prioritized_score
FROM megamind_chunks c 
JOIN megamind_realms r ON c.realm_id = r.realm_id
WHERE c.embedding IS NOT NULL 
    AND r.is_active = TRUE
    AND (c.realm_id = @current_realm OR c.realm_id = 'GLOBAL')
ORDER BY prioritized_score DESC;

-- Performance monitoring view for realm operations
CREATE VIEW realm_performance_metrics AS
SELECT 
    r.realm_id,
    r.realm_name,
    COUNT(DISTINCT sm.session_id) as active_sessions,
    COUNT(sc.change_id) as pending_changes,
    AVG(sc.impact_score) as avg_change_impact,
    COUNT(DISTINCT c.chunk_id) as total_chunks,
    SUM(c.access_count) as total_accesses,
    MAX(c.last_accessed) as last_access_time
FROM megamind_realms r
LEFT JOIN megamind_session_metadata sm ON r.realm_id = sm.realm_id AND sm.is_active = TRUE
LEFT JOIN megamind_session_changes sc ON sm.session_id = sc.session_id AND sc.status = 'pending'
LEFT JOIN megamind_chunks c ON r.realm_id = c.realm_id
WHERE r.is_active = TRUE
GROUP BY r.realm_id, r.realm_name;