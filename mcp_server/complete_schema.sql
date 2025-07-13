-- MegaMind Context Database - Complete Schema
-- GitHub Issue #5: Database Schema Completeness Fix
-- Combines init_schema.sql + init_database.py requirements

USE megamind_database;

-- ================================================================
-- PHASE 1: CORE INFRASTRUCTURE TABLES
-- ================================================================

-- Main chunks table
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
    FULLTEXT(content, section_path),
    FULLTEXT(content)  -- Individual content index for realm-aware search
) ENGINE=InnoDB;

-- Chunk relationships with cross-realm support
CREATE TABLE IF NOT EXISTS megamind_chunk_relationships (
    relationship_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    related_chunk_id VARCHAR(50) NOT NULL,
    relationship_type ENUM('references', 'implements', 'extends', 'uses', 'similar_to') NOT NULL,
    strength DECIMAL(3,2) DEFAULT 0.0,
    discovered_by ENUM('manual', 'semantic', 'structural') DEFAULT 'manual',
    is_cross_realm BOOLEAN DEFAULT FALSE,  -- MISSING COLUMN ADDED
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (related_chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    INDEX idx_chunk_relationships (chunk_id),
    INDEX idx_relationship_type (relationship_type),
    INDEX idx_cross_realm (is_cross_realm)
) ENGINE=InnoDB;

-- Chunk tags
CREATE TABLE IF NOT EXISTS megamind_chunk_tags (
    tag_id VARCHAR(50) PRIMARY KEY,
    chunk_id VARCHAR(50) NOT NULL,
    tag_type ENUM('keyword', 'category', 'priority', 'status', 'language') NOT NULL,
    tag_value VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (chunk_id) REFERENCES megamind_chunks(chunk_id) ON DELETE CASCADE,
    INDEX idx_chunk_tags (chunk_id),
    INDEX idx_tag_type_value (tag_type, tag_value)
) ENGINE=InnoDB;

-- ================================================================
-- PHASE 2: REALM MANAGEMENT TABLES (FROM init_database.py)
-- ================================================================

-- Core realm definitions
CREATE TABLE IF NOT EXISTS megamind_realms (
    realm_id VARCHAR(50) PRIMARY KEY,
    realm_name VARCHAR(255) NOT NULL,
    realm_type ENUM('global', 'project', 'team', 'personal') NOT NULL,
    parent_realm_id VARCHAR(50) DEFAULT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    realm_config JSON DEFAULT NULL,
    created_by VARCHAR(100) DEFAULT 'system',
    access_level ENUM('read_only', 'read_write', 'admin') DEFAULT 'read_write',
    INDEX idx_realms_type (realm_type, is_active),
    INDEX idx_realms_active (is_active, created_at DESC),
    FOREIGN KEY (parent_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE SET NULL
) ENGINE=InnoDB;

-- Realm inheritance hierarchy
CREATE TABLE IF NOT EXISTS megamind_realm_inheritance (
    inheritance_id INT PRIMARY KEY AUTO_INCREMENT,
    child_realm_id VARCHAR(50) NOT NULL,
    parent_realm_id VARCHAR(50) NOT NULL,
    inheritance_type ENUM('full', 'selective', 'read_only') NOT NULL DEFAULT 'full',
    priority_order INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    inheritance_config JSON DEFAULT NULL,
    INDEX idx_inheritance_child (child_realm_id),
    INDEX idx_inheritance_parent (parent_realm_id),
    FOREIGN KEY (child_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE,
    FOREIGN KEY (parent_realm_id) REFERENCES megamind_realms(realm_id) ON DELETE CASCADE
) ENGINE=InnoDB;

-- ================================================================
-- PHASE 3: INTELLIGENCE LAYER TABLES
-- ================================================================

-- Embeddings with realm support
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
) ENGINE=InnoDB;

-- ================================================================
-- PHASE 4: SESSION MANAGEMENT TABLES
-- ================================================================

-- Session metadata
CREATE TABLE IF NOT EXISTS megamind_session_metadata (
    session_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(100),
    project_context VARCHAR(255),
    session_data JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_user_sessions (user_id),
    INDEX idx_last_activity (last_activity DESC)
) ENGINE=InnoDB;

-- Session changes tracking
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
    FOREIGN KEY (session_id) REFERENCES megamind_session_metadata(session_id) ON DELETE CASCADE,
    INDEX idx_session_changes (session_id),
    INDEX idx_change_status (status),
    INDEX idx_impact_score (impact_score DESC)
) ENGINE=InnoDB;

-- Knowledge contributions tracking
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
) ENGINE=InnoDB;

-- ================================================================
-- PHASE 5: MONITORING & OPTIMIZATION TABLES
-- ================================================================

-- Performance metrics
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
) ENGINE=InnoDB;

-- System health monitoring
CREATE TABLE IF NOT EXISTS megamind_system_health (
    health_id VARCHAR(50) PRIMARY KEY,
    component ENUM('database', 'mcp_server', 'analytics', 'review_interface') NOT NULL,
    status ENUM('healthy', 'warning', 'critical', 'down') NOT NULL,
    details JSON,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_component_status (component, status),
    INDEX idx_checked_at (checked_at DESC)
) ENGINE=InnoDB;

-- ================================================================
-- INITIAL DATA POPULATION
-- ================================================================

-- Default realms (GLOBAL prioritized over PROJECT per GitHub Issue #5)
INSERT IGNORE INTO megamind_realms (realm_id, realm_name, realm_type, description, created_by) VALUES
('GLOBAL', 'Global Knowledge Base', 'global', 'Universal knowledge repository - primary source for all queries', 'system'),
('PROJECT', 'Default Project Realm', 'project', 'Project-specific knowledge realm - secondary to GLOBAL', 'system');

-- Sample chunks for testing (inserted into GLOBAL realm for priority)
INSERT IGNORE INTO megamind_chunks (chunk_id, realm_id, content, source_document, section_path, chunk_type, line_count, token_count, access_count) VALUES
('sample_001', 'GLOBAL', 'This is a sample chunk for testing the MegaMind Context Database System. It demonstrates basic functionality and search capabilities.', 'sample_document.md', '/Introduction/Overview', 'section', 2, 25, 1),
('sample_002', 'GLOBAL', 'def search_function(query):\n    return database.search(query)', 'sample_code.py', '/Functions/SearchFunction', 'function', 2, 15, 1),
('sample_003', 'GLOBAL', 'Always use transactions when modifying database records to ensure data consistency and integrity.', 'database_rules.md', '/Rules/Transactions', 'rule', 1, 18, 1);

-- Sample relationships with cross-realm marking
INSERT IGNORE INTO megamind_chunk_relationships (relationship_id, chunk_id, related_chunk_id, relationship_type, strength, is_cross_realm) VALUES
('rel_001', 'sample_001', 'sample_002', 'references', 0.8, FALSE),
('rel_002', 'sample_002', 'sample_003', 'implements', 0.9, FALSE);

-- Sample tags
INSERT IGNORE INTO megamind_chunk_tags (tag_id, chunk_id, tag_type, tag_value) VALUES
('tag_001', 'sample_001', 'category', 'introduction'),
('tag_002', 'sample_002', 'language', 'python'),
('tag_003', 'sample_003', 'priority', 'high');

-- ================================================================
-- STORED FUNCTIONS FOR INHERITANCE RESOLUTION (GitHub Issue #5 Phase 3)
-- ================================================================

-- Drop existing function if exists
DROP FUNCTION IF EXISTS resolve_inheritance_conflict;

-- Enhanced inheritance conflict resolution function
DELIMITER $$
CREATE FUNCTION resolve_inheritance_conflict(
    p_chunk_id VARCHAR(50), 
    p_accessing_realm VARCHAR(50)
) 
RETURNS JSON
READS SQL DATA
DETERMINISTIC
BEGIN
    DECLARE result JSON;
    DECLARE v_chunk_realm VARCHAR(50) DEFAULT NULL;
    DECLARE v_inheritance_type VARCHAR(20) DEFAULT NULL;
    DECLARE v_priority_order INT DEFAULT NULL;
    DECLARE v_has_inheritance BOOLEAN DEFAULT FALSE;
    
    -- Get chunk's realm
    SELECT realm_id INTO v_chunk_realm
    FROM megamind_chunks 
    WHERE chunk_id = p_chunk_id;
    
    -- Return failure if chunk doesn't exist
    IF v_chunk_realm IS NULL THEN
        SET result = JSON_OBJECT(
            'access_granted', FALSE,
            'access_type', 'denied',
            'reason', 'Chunk not found',
            'chunk_id', p_chunk_id
        );
        RETURN result;
    END IF;
    
    -- Direct access (same realm)
    IF v_chunk_realm = p_accessing_realm THEN
        SET result = JSON_OBJECT(
            'access_granted', TRUE,
            'access_type', 'direct',
            'source_realm', p_accessing_realm,
            'target_realm', v_chunk_realm,
            'priority_order', 1,
            'reason', 'Direct access to own realm'
        );
        RETURN result;
    END IF;
    
    -- GLOBAL realm access (always accessible to all realms)
    IF v_chunk_realm = 'GLOBAL' THEN
        SET result = JSON_OBJECT(
            'access_granted', TRUE,
            'access_type', 'inherited',
            'source_realm', 'GLOBAL',
            'target_realm', p_accessing_realm,
            'priority_order', 10,
            'inheritance_type', 'global_access',
            'reason', 'Global realm inheritance - accessible to all'
        );
        RETURN result;
    END IF;
    
    -- Check for explicit inheritance relationship
    SELECT 
        inheritance_type, 
        priority_order,
        TRUE
    INTO 
        v_inheritance_type, 
        v_priority_order,
        v_has_inheritance
    FROM megamind_realm_inheritance 
    WHERE child_realm_id = p_accessing_realm 
      AND parent_realm_id = v_chunk_realm 
      AND is_active = TRUE
    ORDER BY priority_order ASC
    LIMIT 1;
    
    -- Grant access if inheritance relationship exists
    IF v_has_inheritance THEN
        SET result = JSON_OBJECT(
            'access_granted', TRUE,
            'access_type', 'inherited',
            'source_realm', v_chunk_realm,
            'target_realm', p_accessing_realm,
            'priority_order', v_priority_order,
            'inheritance_type', v_inheritance_type,
            'reason', CONCAT('Inherited access via ', v_inheritance_type, ' inheritance')
        );
        RETURN result;
    END IF;
    
    -- Check reverse inheritance (parent accessing child)
    SELECT 
        inheritance_type, 
        priority_order,
        TRUE
    INTO 
        v_inheritance_type, 
        v_priority_order,
        v_has_inheritance
    FROM megamind_realm_inheritance 
    WHERE child_realm_id = v_chunk_realm 
      AND parent_realm_id = p_accessing_realm 
      AND is_active = TRUE
      AND inheritance_type IN ('full', 'selective')
    ORDER BY priority_order ASC
    LIMIT 1;
    
    -- Grant reverse access if allowed
    IF v_has_inheritance THEN
        SET result = JSON_OBJECT(
            'access_granted', TRUE,
            'access_type', 'reverse_inherited',
            'source_realm', p_accessing_realm,
            'target_realm', v_chunk_realm,
            'priority_order', v_priority_order + 5,
            'inheritance_type', v_inheritance_type,
            'reason', CONCAT('Reverse inherited access via parent realm')
        );
        RETURN result;
    END IF;
    
    -- Deny access - no inheritance path found
    SET result = JSON_OBJECT(
        'access_granted', FALSE,
        'access_type', 'denied',
        'source_realm', p_accessing_realm,
        'target_realm', v_chunk_realm,
        'reason', 'No inheritance path found between realms'
    );
    
    RETURN result;
END$$
DELIMITER ;

-- ================================================================
-- SCHEMA VALIDATION QUERIES
-- ================================================================

-- Verify table completeness (for health checks)
SELECT 
    'Schema Validation' as check_type,
    COUNT(*) as table_count,
    'Expected: 10 tables' as expected
FROM information_schema.tables 
WHERE table_schema = 'megamind_database' 
AND table_name LIKE 'megamind_%';

-- Verify sample data exists
SELECT 
    'Sample Data Validation' as check_type,
    COUNT(*) as chunk_count,
    'Expected: 3 chunks in GLOBAL realm' as expected
FROM megamind_chunks 
WHERE realm_id = 'GLOBAL';

-- Verify stored functions exist
SELECT 
    'Function Validation' as check_type,
    COUNT(*) as function_count,
    'Expected: 1 function (resolve_inheritance_conflict)' as expected
FROM information_schema.routines 
WHERE routine_schema = 'megamind_database' 
AND routine_name = 'resolve_inheritance_conflict';